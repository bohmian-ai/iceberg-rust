use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::spec::{Manifest, ManifestEntry, ManifestFile, ManifestStatus, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, Transaction, TransactionAction};
use crate::{Catalog, Error, ErrorKind, Result};

/// Exact manifest identities selected for compatible rewriting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestRewriteSelection {
    /// Manifest paths that must all belong to the current snapshot.
    pub manifest_paths: Vec<String>,
}

/// Bounds a manifest rewrite before it writes replacement metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestRewriteLimits {
    /// Maximum selected manifests.
    pub max_manifests: usize,
    /// Maximum entries read from selected manifests.
    pub max_entries: usize,
    /// Maximum declared selected-manifest bytes.
    pub max_bytes: u64,
}

/// Classification of a bounded rewrite request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestRewriteOutcome {
    /// No manifest was selected, so no catalog commit was attempted.
    NoOp,
    /// Selected identities no longer match the table's current snapshot.
    Stale,
    /// Compatible replacement manifests were committed.
    Rewritten,
}

/// Exact result of a manifest rewrite attempt.
#[derive(Clone)]
pub struct ManifestRewriteResult {
    /// Exact table returned by the successful commit, or the input table for no-op/stale.
    pub table: Table,
    /// Selected manifest identities replaced by the commit.
    pub rewritten_manifest_paths: Vec<String>,
    /// Whether the request was committed, stale, or a no-op.
    pub outcome: ManifestRewriteOutcome,
}

struct RewriteManifestsAction {
    selected: HashSet<String>,
    limits: ManifestRewriteLimits,
    commit_uuid: Uuid,
}

#[async_trait]
impl TransactionAction for RewriteManifestsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let producer = SnapshotProducer::new(
            table,
            self.commit_uuid,
            None,
            HashMap::from([("manifest-rewrite".to_string(), "true".to_string())]),
            Vec::new(),
        );
        producer
            .commit(
                RewriteManifestOperation {
                    selected: self.selected.clone(),
                    limits: self.limits,
                },
                DefaultManifestProcess,
            )
            .await
    }
}

struct RewriteManifestOperation {
    selected: HashSet<String>,
    limits: ManifestRewriteLimits,
}

async fn selected_manifests(
    operation: &RewriteManifestOperation,
    producer: &SnapshotProducer<'_>,
) -> Result<Vec<(ManifestFile, Manifest)>> {
    let Some(snapshot) = producer.table.metadata().current_snapshot() else {
        return Ok(Vec::new());
    };
    let list = producer.table.manifest_list_reader(snapshot).load().await?;
    let available: HashSet<&str> = list
        .entries()
        .iter()
        .map(|manifest| manifest.manifest_path.as_str())
        .collect();
    if !operation
        .selected
        .iter()
        .all(|path| available.contains(path.as_str()))
    {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Selected manifest is not in the current snapshot",
        ));
    }
    if operation.selected.len() > operation.limits.max_manifests {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Manifest rewrite count limit exhausted",
        ));
    }
    let mut bytes = 0_u64;
    let mut entries = 0_usize;
    let mut selected = Vec::new();
    for file in list
        .entries()
        .iter()
        .filter(|file| operation.selected.contains(&file.manifest_path))
    {
        bytes =
            bytes
                .checked_add(u64::try_from(file.manifest_length).map_err(|_| {
                    Error::new(ErrorKind::DataInvalid, "Manifest length is negative")
                })?)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Manifest rewrite byte limit exhausted",
                    )
                })?;
        let manifest = file.load_manifest(producer.table.file_io()).await?;
        entries = entries
            .checked_add(manifest.entries().len())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Manifest rewrite entry limit exhausted",
                )
            })?;
        if bytes > operation.limits.max_bytes || entries > operation.limits.max_entries {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Manifest rewrite limit exhausted",
            ));
        }
        selected.push((file.clone(), manifest));
    }
    Ok(selected)
}

impl SnapshotProduceOperation for RewriteManifestOperation {
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    fn rewrite_entries(&self) -> bool {
        true
    }

    async fn delete_entries(&self, producer: &SnapshotProducer<'_>) -> Result<Vec<ManifestEntry>> {
        let manifests = selected_manifests(self, producer).await?;
        Ok(manifests
            .into_iter()
            .flat_map(|(_, manifest)| manifest.entries().to_vec())
            .filter(|entry| entry.is_alive())
            .map(|entry| {
                ManifestEntry::builder()
                    .status(ManifestStatus::Existing)
                    .snapshot_id(producer.snapshot_id())
                    .sequence_number(entry.sequence_number().unwrap_or(0))
                    .file_sequence_number_opt(entry.file_sequence_number)
                    .data_file(entry.data_file().clone())
                    .build()
            })
            .collect())
    }

    async fn existing_manifest(
        &self,
        producer: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let Some(snapshot) = producer.table.metadata().current_snapshot() else {
            return Ok(Vec::new());
        };
        let list = producer.table.manifest_list_reader(snapshot).load().await?;
        Ok(list
            .entries()
            .iter()
            .filter(|file| !self.selected.contains(&file.manifest_path))
            .cloned()
            .collect())
    }
}

/// Rewrites exact current-snapshot manifests without changing their live file set.
///
/// # Errors
///
/// Returns an error when limits are empty or exhausted, selected content is malformed,
/// metadata IO fails, or the single catalog commit fails.
pub async fn rewrite_manifests(
    catalog: &dyn Catalog,
    table: &Table,
    selection: ManifestRewriteSelection,
    limits: ManifestRewriteLimits,
) -> Result<ManifestRewriteResult> {
    let mut paths = selection.manifest_paths;
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        return Ok(ManifestRewriteResult {
            table: table.clone(),
            rewritten_manifest_paths: paths,
            outcome: ManifestRewriteOutcome::NoOp,
        });
    }
    if limits.max_manifests == 0 || limits.max_entries == 0 || limits.max_bytes == 0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Manifest rewrite limits must be positive",
        ));
    }
    let current = catalog.load_table(table.identifier()).await?;
    if current.metadata() != table.metadata()
        || current.metadata_location() != table.metadata_location()
    {
        return Ok(ManifestRewriteResult {
            table: current,
            rewritten_manifest_paths: Vec::new(),
            outcome: ManifestRewriteOutcome::Stale,
        });
    }
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(ManifestRewriteResult {
            table: table.clone(),
            rewritten_manifest_paths: Vec::new(),
            outcome: ManifestRewriteOutcome::Stale,
        });
    };
    let list = table.manifest_list_reader(snapshot).load().await?;
    let available: HashSet<&str> = list
        .entries()
        .iter()
        .map(|file| file.manifest_path.as_str())
        .collect();
    if !paths.iter().all(|path| available.contains(path.as_str())) {
        return Ok(ManifestRewriteResult {
            table: table.clone(),
            rewritten_manifest_paths: Vec::new(),
            outcome: ManifestRewriteOutcome::Stale,
        });
    }
    let selected = paths.iter().cloned().collect();
    let action = RewriteManifestsAction {
        selected,
        limits,
        commit_uuid: Uuid::now_v7(),
    };
    let mut transaction = Transaction::new(table);
    transaction.actions.push(Arc::new(action));
    let committed = transaction.commit_once(catalog).await?;
    Ok(ManifestRewriteResult {
        table: committed,
        rewritten_manifest_paths: paths,
        outcome: ManifestRewriteOutcome::Rewritten,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::MockCatalog;
    use crate::transaction::tests::make_v2_minimal_table;

    #[tokio::test]
    async fn rewrite_manifests_collapses_empty_selection_without_commit() {
        let table = make_v2_minimal_table();
        let result = rewrite_manifests(
            &MockCatalog::new(),
            &table,
            ManifestRewriteSelection {
                manifest_paths: vec![],
            },
            ManifestRewriteLimits {
                max_manifests: 1,
                max_entries: 1,
                max_bytes: 1,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.outcome, ManifestRewriteOutcome::NoOp);
        assert!(result.rewritten_manifest_paths.is_empty());
        assert_eq!(result.table.metadata(), table.metadata());
    }
}
