use std::collections::HashSet;

use crate::io::FileIO;
use crate::spec::{DataContentType, TableMetadata};
use crate::{Error, ErrorKind, Result};

/// Bounds metadata traversal before any cleanup candidates are returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CleanupTraversalLimits {
    /// Maximum distinct or repeated metadata and content records visited.
    pub max_items: usize,
    /// Maximum declared manifest bytes plus path bytes visited.
    pub max_bytes: u64,
}

/// Typed paths that became unreachable after snapshot expiration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpiredFileSet {
    /// Data-file paths no longer reachable from table metadata.
    pub data_files: HashSet<String>,
    /// Delete-file paths no longer reachable from table metadata.
    pub delete_files: HashSet<String>,
    /// Manifest paths no longer reachable from table metadata.
    pub manifests: HashSet<String>,
    /// Manifest-list paths no longer reachable from table metadata.
    pub manifest_lists: HashSet<String>,
    /// Statistics and partition-statistics paths no longer reachable.
    pub statistics: HashSet<String>,
    /// Metadata-log paths no longer reachable from table metadata.
    pub metadata_logs: HashSet<String>,
    /// Whether traversal stopped at a limit. Successful traversal is always complete.
    pub limit_exhausted: bool,
}

#[derive(Default)]
struct ReachableFiles {
    files: ExpiredFileSet,
}

struct TraversalBudget {
    remaining_items: usize,
    remaining_bytes: u64,
}

impl TraversalBudget {
    fn consume(&mut self, path: &str, declared_bytes: u64) -> bool {
        let Some(path_bytes) = u64::try_from(path.len()).ok() else {
            return false;
        };
        let Some(bytes) = declared_bytes.checked_add(path_bytes) else {
            return false;
        };
        if self.remaining_items == 0 || bytes > self.remaining_bytes {
            return false;
        }
        self.remaining_items -= 1;
        self.remaining_bytes -= bytes;
        true
    }
}

fn validate_lineage(before: &TableMetadata, after: &TableMetadata) -> Result<()> {
    if before.uuid() != after.uuid() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Cleanup metadata belongs to different tables",
        ));
    }
    for snapshot in after.snapshots() {
        let Some(previous) = before.snapshot_by_id(snapshot.snapshot_id()) else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Post-expiration metadata contains a new snapshot",
            ));
        };
        if previous != snapshot {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Post-expiration snapshot changed during cleanup derivation",
            ));
        }
    }
    for (name, reference) in &after.refs {
        if before.refs.get(name) != Some(reference) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Post-expiration reference is not retained from before metadata",
            ));
        }
    }
    Ok(())
}

async fn reachable(
    file_io: &FileIO,
    metadata: &TableMetadata,
    budget: &mut TraversalBudget,
) -> Result<Option<ReachableFiles>> {
    let mut reachable = ReachableFiles::default();
    for log in metadata.metadata_log() {
        if !budget.consume(&log.metadata_file, 0) {
            return Ok(None);
        }
        reachable
            .files
            .metadata_logs
            .insert(log.metadata_file.clone());
    }
    for stats in metadata.statistics_iter() {
        let size = u64::try_from(stats.file_size_in_bytes)
            .map_err(|_| Error::new(ErrorKind::DataInvalid, "Statistics file size is negative"))?;
        if !budget.consume(&stats.statistics_path, size) {
            return Ok(None);
        }
        reachable
            .files
            .statistics
            .insert(stats.statistics_path.clone());
    }
    for stats in metadata.partition_statistics_iter() {
        let size = u64::try_from(stats.file_size_in_bytes).map_err(|_| {
            Error::new(
                ErrorKind::DataInvalid,
                "Partition statistics file size is negative",
            )
        })?;
        if !budget.consume(&stats.statistics_path, size) {
            return Ok(None);
        }
        reachable
            .files
            .statistics
            .insert(stats.statistics_path.clone());
    }
    for snapshot in metadata.snapshots() {
        if !budget.consume(snapshot.manifest_list(), 0) {
            return Ok(None);
        }
        reachable
            .files
            .manifest_lists
            .insert(snapshot.manifest_list().to_string());
        let bytes = file_io.new_input(snapshot.manifest_list())?.read().await?;
        let list =
            crate::spec::ManifestList::parse_with_version(&bytes, metadata.format_version())?;
        for manifest_file in list.entries() {
            let declared = u64::try_from(manifest_file.manifest_length)
                .map_err(|_| Error::new(ErrorKind::DataInvalid, "Manifest length is negative"))?;
            if !budget.consume(&manifest_file.manifest_path, declared) {
                return Ok(None);
            }
            reachable
                .files
                .manifests
                .insert(manifest_file.manifest_path.clone());
            let manifest = manifest_file.load_manifest(file_io).await?;
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                if !budget.consume(entry.file_path(), entry.data_file().file_size_in_bytes()) {
                    return Ok(None);
                }
                match entry.data_file().content_type() {
                    DataContentType::Data => {
                        reachable
                            .files
                            .data_files
                            .insert(entry.file_path().to_string());
                    }
                    DataContentType::PositionDeletes | DataContentType::EqualityDeletes => {
                        reachable
                            .files
                            .delete_files
                            .insert(entry.file_path().to_string());
                    }
                }
            }
        }
    }
    Ok(Some(reachable))
}

/// Derives files reachable before, but not after, a metadata-only expiration commit.
///
/// The operation performs only `FileIO` reads. It never deletes an object.
///
/// # Errors
///
/// Returns an error for incompatible lineage, malformed metadata content, unreadable
/// manifests, or invalid sizes. Exhaustion returns a typed set with
/// `limit_exhausted = true` and no actionable paths.
pub async fn expired_files_between(
    file_io: &FileIO,
    before: &TableMetadata,
    after: &TableMetadata,
    limits: CleanupTraversalLimits,
) -> Result<ExpiredFileSet> {
    validate_lineage(before, after)?;
    let mut budget = TraversalBudget {
        remaining_items: limits.max_items,
        remaining_bytes: limits.max_bytes,
    };
    let Some(before) = reachable(file_io, before, &mut budget).await? else {
        return Ok(ExpiredFileSet {
            limit_exhausted: true,
            ..ExpiredFileSet::default()
        });
    };
    let Some(after) = reachable(file_io, after, &mut budget).await? else {
        return Ok(ExpiredFileSet {
            limit_exhausted: true,
            ..ExpiredFileSet::default()
        });
    };
    let before = before.files;
    let after = after.files;
    Ok(ExpiredFileSet {
        data_files: &before.data_files - &after.data_files,
        delete_files: &before.delete_files - &after.delete_files,
        manifests: &before.manifests - &after.manifests,
        manifest_lists: &before.manifest_lists - &after.manifest_lists,
        statistics: &before.statistics - &after.statistics,
        metadata_logs: &before.metadata_logs - &after.metadata_logs,
        limit_exhausted: false,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestEntry,
        ManifestListWriter, ManifestStatus, ManifestWriterBuilder, MetadataLog, StatisticsFile,
        Struct,
    };
    use crate::table::Table;
    use crate::transaction::append::tests::make_table_with_delete_only_manifest;
    use crate::transaction::tests::{make_v2_minimal_table, make_v2_table};

    #[tokio::test]
    async fn expired_files_rejects_invalid_lineage_before_io() {
        let table = make_v2_table();
        let after = table
            .metadata()
            .clone()
            .into_builder(None)
            .assign_uuid(Uuid::now_v7())
            .build()
            .unwrap()
            .metadata;
        let error = expired_files_between(
            table.file_io(),
            table.metadata(),
            &after,
            CleanupTraversalLimits {
                max_items: 1,
                max_bytes: 1,
            },
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    #[tokio::test]
    async fn expired_files_reports_zero_limit_exhaustion_without_candidates() {
        let table = make_v2_table();
        let expired = expired_files_between(
            table.file_io(),
            table.metadata(),
            table.metadata(),
            CleanupTraversalLimits {
                max_items: 0,
                max_bytes: 0,
            },
        )
        .await
        .unwrap();
        assert!(expired.limit_exhausted);
        assert!(expired.data_files.is_empty());
        assert!(expired.delete_files.is_empty());
        assert!(expired.manifests.is_empty());
        assert!(expired.manifest_lists.is_empty());
        assert!(expired.statistics.is_empty());
        assert!(expired.metadata_logs.is_empty());
    }

    #[tokio::test]
    async fn expired_files_shares_nonzero_item_budget_across_both_traversals() {
        let table = make_v2_minimal_table();
        let mut metadata = table.metadata().clone();
        metadata.snapshots.clear();
        metadata.current_snapshot_id = None;
        metadata.refs.clear();
        metadata.statistics.clear();
        metadata.partition_statistics.clear();
        metadata.metadata_log = vec![MetadataLog {
            metadata_file: "memory://metadata/previous.json".to_string(),
            timestamp_ms: 1,
        }];
        let per_side_items = metadata.metadata_log().len();
        assert!(per_side_items > 0);
        let expired = expired_files_between(
            table.file_io(),
            &metadata,
            &metadata,
            CleanupTraversalLimits {
                max_items: per_side_items,
                max_bytes: u64::MAX,
            },
        )
        .await
        .unwrap();
        assert!(expired.limit_exhausted);
        assert!(expired.data_files.is_empty());
    }

    async fn cleanup_fixture() -> (Table, TempDir, TableMetadata, TableMetadata) {
        let (table, temp_dir, _) = make_table_with_delete_only_manifest().await;
        let current = table.metadata().current_snapshot().unwrap();
        let original_list = table.manifest_list_reader(current).load().await.unwrap();
        let schema = current.schema(table.metadata()).unwrap();
        let spec = table.metadata().default_partition_spec().as_ref().clone();
        let delete_manifest_path =
            format!("{}/metadata/live-delete.avro", table.metadata().location());
        let mut delete_writer = ManifestWriterBuilder::new(
            table.file_io().new_output(delete_manifest_path).unwrap(),
            Some(current.snapshot_id()),
            None,
            schema,
            spec,
        )
        .build_v2_deletes();
        delete_writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .snapshot_id(current.snapshot_id())
                    .sequence_number(1)
                    .file_sequence_number(1)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::PositionDeletes)
                            .file_path(format!("{}/delete.parquet", table.metadata().location()))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(40)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let delete_manifest = delete_writer.write_manifest_file().await.unwrap();

        let old_list_path = format!("{}/metadata/expired-list.avro", table.metadata().location());
        let mut old_list_writer = ManifestListWriter::v2(
            table
                .file_io()
                .new_output(&old_list_path)
                .unwrap()
                .writer()
                .await
                .unwrap(),
            current.snapshot_id(),
            current.parent_snapshot_id(),
            current.sequence_number(),
        );
        old_list_writer
            .add_manifests(
                original_list
                    .entries()
                    .iter()
                    .cloned()
                    .chain([delete_manifest]),
            )
            .unwrap();
        old_list_writer.close().await.unwrap();

        let empty_list_path = format!(
            "{}/metadata/current-empty-list.avro",
            table.metadata().location()
        );
        let mut empty_writer = ManifestListWriter::v2(
            table
                .file_io()
                .new_output(&empty_list_path)
                .unwrap()
                .writer()
                .await
                .unwrap(),
            current.snapshot_id(),
            current.parent_snapshot_id(),
            current.sequence_number(),
        );
        empty_writer.add_manifests(std::iter::empty()).unwrap();
        empty_writer.close().await.unwrap();

        let mut json = serde_json::to_value(table.metadata()).unwrap();
        let snapshots = json.get_mut("snapshots").unwrap().as_array_mut().unwrap();
        for snapshot in snapshots {
            let path = if snapshot["snapshot-id"] == 3051729675574597004_i64 {
                old_list_path.clone()
            } else {
                empty_list_path.clone()
            };
            snapshot["manifest-list"] = serde_json::Value::String(path);
        }
        json["statistics"] = serde_json::json!([{
            "snapshot-id": 3051729675574597004_i64,
            "statistics-path": format!("{}/old.stats", table.metadata().location()),
            "file-size-in-bytes": 20,
            "file-footer-size-in-bytes": 1,
            "blob-metadata": []
        }]);
        let before: TableMetadata = serde_json::from_value(json).unwrap();
        let mut after = before.clone();
        after.snapshots.remove(&3051729675574597004_i64);
        after.refs.clear();
        after.statistics.remove(&3051729675574597004_i64);
        after.metadata_log.clear();
        (table, temp_dir, before, after)
    }

    #[tokio::test]
    async fn expired_files_classifies_fixture_paths_and_collapses_duplicates() {
        let (table, _temp_dir, before, after) = cleanup_fixture().await;
        let expired =
            expired_files_between(table.file_io(), &before, &after, CleanupTraversalLimits {
                max_items: 100,
                max_bytes: u64::MAX,
            })
            .await
            .unwrap();
        assert!(!expired.limit_exhausted);
        assert_eq!(expired.data_files.len(), 1);
        assert_eq!(expired.delete_files.len(), 1);
        assert_eq!(expired.manifest_lists.len(), 1);
        assert_eq!(expired.statistics.len(), 1);
        assert_eq!(expired.metadata_logs.len(), 1);
        assert_eq!(expired.manifests.len(), 3);
    }

    #[tokio::test]
    async fn expired_files_rejects_negative_statistics_size() {
        let table = make_v2_minimal_table();
        let mut before = table.metadata().clone();
        before.statistics.insert(1, StatisticsFile {
            snapshot_id: 1,
            statistics_path: "memory://negative.stats".to_string(),
            file_size_in_bytes: -1,
            file_footer_size_in_bytes: 0,
            key_metadata: None,
            blob_metadata: Vec::new(),
        });
        let error = expired_files_between(
            table.file_io(),
            &before,
            table.metadata(),
            CleanupTraversalLimits {
                max_items: 10,
                max_bytes: 10,
            },
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    #[tokio::test]
    async fn expired_files_protects_retained_refs_and_shared_paths() {
        let (table, _temp_dir, _) = make_table_with_delete_only_manifest().await;
        assert!(!table.metadata().refs.is_empty());
        let expired = expired_files_between(
            table.file_io(),
            table.metadata(),
            table.metadata(),
            CleanupTraversalLimits {
                max_items: 100,
                max_bytes: u64::MAX,
            },
        )
        .await
        .unwrap();
        assert!(!expired.limit_exhausted);
        assert_eq!(expired, ExpiredFileSet::default());
    }

    #[tokio::test]
    async fn expired_files_reports_nonzero_byte_exhaustion_without_candidates() {
        let (table, _temp_dir, _) = make_table_with_delete_only_manifest().await;
        let expired = expired_files_between(
            table.file_io(),
            table.metadata(),
            table.metadata(),
            CleanupTraversalLimits {
                max_items: 100,
                max_bytes: 1,
            },
        )
        .await
        .unwrap();
        assert!(expired.limit_exhausted);
        assert!(expired.manifests.is_empty());
    }

    #[tokio::test]
    async fn expired_files_rejects_malformed_manifest_list_content() {
        let (table, _temp_dir, _) = make_table_with_delete_only_manifest().await;
        fs::write(
            table.metadata().current_snapshot().unwrap().manifest_list(),
            b"not avro",
        )
        .unwrap();
        let error = expired_files_between(
            table.file_io(),
            table.metadata(),
            table.metadata(),
            CleanupTraversalLimits {
                max_items: 100,
                max_bytes: u64::MAX,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            ErrorKind::DataInvalid | ErrorKind::Unexpected
        ));
    }
}
