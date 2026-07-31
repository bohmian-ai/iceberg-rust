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
    items: usize,
    bytes: u64,
}

impl ReachableFiles {
    fn account(
        &mut self,
        path: &str,
        declared_bytes: u64,
        limits: CleanupTraversalLimits,
    ) -> Result<()> {
        self.items = self.items.checked_add(1).ok_or_else(limit_error)?;
        self.bytes = self
            .bytes
            .checked_add(declared_bytes)
            .and_then(|n| n.checked_add(path.len() as u64))
            .ok_or_else(limit_error)?;
        if self.items > limits.max_items || self.bytes > limits.max_bytes {
            return Err(limit_error());
        }
        Ok(())
    }
}

fn limit_error() -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        "Cleanup metadata traversal limit exhausted",
    )
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
    limits: CleanupTraversalLimits,
) -> Result<ReachableFiles> {
    let mut reachable = ReachableFiles::default();
    for log in metadata.metadata_log() {
        reachable.account(&log.metadata_file, 0, limits)?;
        reachable
            .files
            .metadata_logs
            .insert(log.metadata_file.clone());
    }
    for stats in metadata.statistics_iter() {
        let size = u64::try_from(stats.file_size_in_bytes)
            .map_err(|_| Error::new(ErrorKind::DataInvalid, "Statistics file size is negative"))?;
        reachable.account(&stats.statistics_path, size, limits)?;
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
        reachable.account(&stats.statistics_path, size, limits)?;
        reachable
            .files
            .statistics
            .insert(stats.statistics_path.clone());
    }
    for snapshot in metadata.snapshots() {
        reachable.account(snapshot.manifest_list(), 0, limits)?;
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
            reachable.account(&manifest_file.manifest_path, declared, limits)?;
            reachable
                .files
                .manifests
                .insert(manifest_file.manifest_path.clone());
            let manifest = manifest_file.load_manifest(file_io).await?;
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                reachable.account(
                    entry.file_path(),
                    entry.data_file().file_size_in_bytes(),
                    limits,
                )?;
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
    Ok(reachable)
}

/// Derives files reachable before, but not after, a metadata-only expiration commit.
///
/// The operation performs only `FileIO` reads. It never deletes an object.
///
/// # Errors
///
/// Returns an error for incompatible lineage, malformed metadata content, unreadable
/// manifests, invalid sizes, or traversal limit exhaustion.
pub async fn expired_files_between(
    file_io: &FileIO,
    before: &TableMetadata,
    after: &TableMetadata,
    limits: CleanupTraversalLimits,
) -> Result<ExpiredFileSet> {
    validate_lineage(before, after)?;
    if limits.max_items == 0 || limits.max_bytes == 0 {
        return Err(limit_error());
    }
    let before = reachable(file_io, before, limits).await?.files;
    let after = reachable(file_io, after, limits).await?.files;
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
    use uuid::Uuid;

    use super::*;
    use crate::transaction::tests::make_v2_minimal_table;

    #[tokio::test]
    async fn expired_files_rejects_invalid_lineage_before_io() {
        let table = make_v2_minimal_table();
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
    async fn expired_files_rejects_zero_limits_before_io() {
        let table = make_v2_minimal_table();
        let error = expired_files_between(
            table.file_io(),
            table.metadata(),
            table.metadata(),
            CleanupTraversalLimits {
                max_items: 0,
                max_bytes: 0,
            },
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }
}
