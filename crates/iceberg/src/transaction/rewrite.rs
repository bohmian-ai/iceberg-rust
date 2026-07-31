use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{DataFile, Manifest, ManifestEntry, ManifestFile, ManifestStatus, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// A copy-on-write REPLACE action that removes selected live data files and
/// adds their replacement files in one Iceberg snapshot.
pub struct RewriteFilesAction {
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    removed_file_paths: Vec<String>,
}

impl RewriteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            removed_file_paths: vec![],
        }
    }

    /// Add replacement data files to the new snapshot.
    pub fn add_data_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(files);
        self
    }

    /// Remove these live data-file paths from the new snapshot.
    pub fn delete_files(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        self.removed_file_paths.extend(paths);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = properties;
        self
    }

    /// Set the commit UUID used for generated manifest paths.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set key metadata for generated manifest files.
    pub fn set_key_metadata(mut self, key_metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(key_metadata);
        self
    }
}

#[async_trait]
impl TransactionAction for RewriteFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
        );
        snapshot_producer.validate_added_data_files()?;
        snapshot_producer
            .commit(
                RewriteOperation {
                    removed: self.removed_file_paths.iter().cloned().collect(),
                },
                DefaultManifestProcess,
            )
            .await
    }
}

struct RewriteOperation {
    removed: HashSet<String>,
}

fn manifest_touches(manifest: &Manifest, removed: &HashSet<String>) -> bool {
    manifest
        .entries()
        .iter()
        .any(|entry| entry.is_alive() && removed.contains(entry.file_path()))
}

impl SnapshotProduceOperation for RewriteOperation {
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    async fn delete_entries(&self, producer: &SnapshotProducer<'_>) -> Result<Vec<ManifestEntry>> {
        let Some(current) = producer.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };
        let manifest_list = producer.table.manifest_list_reader(current).load().await?;
        let mut rewritten = Vec::new();
        let mut removed_found = 0_usize;

        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file
                .load_manifest(producer.table.file_io())
                .await?;
            if !manifest_touches(&manifest, &self.removed) {
                continue;
            }
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let status = if self.removed.contains(entry.file_path()) {
                    removed_found += 1;
                    ManifestStatus::Deleted
                } else {
                    ManifestStatus::Existing
                };
                rewritten.push(
                    ManifestEntry::builder()
                        .status(status)
                        .snapshot_id(producer.snapshot_id())
                        .sequence_number(entry.sequence_number().unwrap_or(0))
                        .file_sequence_number_opt(entry.file_sequence_number)
                        .data_file(entry.data_file().clone())
                        .build(),
                );
            }
        }

        if removed_found != self.removed.len() {
            return Err(crate::Error::new(
                crate::ErrorKind::DataInvalid,
                "REPLACE target files are not all present in the current snapshot",
            ));
        }
        Ok(rewritten)
    }

    async fn existing_manifest(
        &self,
        producer: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let Some(snapshot) = producer.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };
        let manifest_list = producer.table.manifest_list_reader(snapshot).load().await?;
        let mut kept = Vec::new();
        for manifest_file in manifest_list.entries() {
            if !(manifest_file.has_added_files()
                || manifest_file.has_existing_files()
                || manifest_file.has_deleted_files())
            {
                continue;
            }
            let manifest = manifest_file
                .load_manifest(producer.table.file_io())
                .await?;
            if !manifest_touches(&manifest, &self.removed) {
                kept.push(manifest_file.clone());
            }
        }
        Ok(kept)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::TableUpdate;
    use crate::spec::SnapshotRef;
    use crate::transaction::TransactionAction;
    use crate::transaction::append::tests::make_table_with_delete_only_manifest;

    #[tokio::test]
    async fn rewrite_files_action_does_not_materialize_manifest_rewrite_entries() {
        let (table, _temp_dir, delete_only_path) = make_table_with_delete_only_manifest().await;
        let snapshot = table.metadata().current_snapshot().unwrap();
        let list = table.manifest_list_reader(snapshot).load().await.unwrap();
        let data_manifest = list
            .entries()
            .iter()
            .find(|m| m.manifest_path != delete_only_path)
            .unwrap();
        let data_path = data_manifest
            .load_manifest(table.file_io())
            .await
            .unwrap()
            .entries()[0]
            .file_path()
            .to_string();
        let action = RewriteFilesAction::new()
            .delete_files([data_path])
            .set_snapshot_properties(HashMap::from([(
                "rewrite-files-regression".to_string(),
                "true".to_string(),
            )]));
        let mut commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = commit.take_updates();
        let new_snapshot = updates
            .iter()
            .find_map(|update| match update {
                TableUpdate::AddSnapshot { snapshot } => Some(SnapshotRef::new(snapshot.clone())),
                _ => None,
            })
            .unwrap();
        let rewritten_list = table
            .manifest_list_reader(&new_snapshot)
            .load()
            .await
            .unwrap();
        assert_eq!(rewritten_list.entries().len(), 1);
        assert_eq!(rewritten_list.entries()[0].manifest_path, delete_only_path);
    }
}
