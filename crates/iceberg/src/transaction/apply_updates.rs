// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::table::Table;
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{TableRequirement, TableUpdate};

/// Transaction action that commits caller-built metadata updates as given.
///
/// This is the general form of the typed actions: it carries any
/// [`TableUpdate`]s — for example `AddSchema`/`SetCurrentSchema` for a rename
/// that keeps field ids, `AddSpec`/`SetDefaultSpec` for partition evolution, or
/// `AddSortOrder`/`SetDefaultSortOrder` — with the [`TableRequirement`]s that
/// must still hold. Requirements are re-checked against the table the commit is
/// applied to, so a concurrent change fails the commit instead of being
/// overwritten.
#[derive(Debug, Default)]
pub struct ApplyUpdatesAction {
    updates: Vec<TableUpdate>,
    requirements: Vec<TableRequirement>,
}

impl ApplyUpdatesAction {
    /// Creates an empty action.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one metadata update, applied in insertion order.
    pub fn update(mut self, update: TableUpdate) -> Self {
        self.updates.push(update);
        self
    }

    /// Appends one requirement the table must meet at commit time.
    pub fn requirement(mut self, requirement: TableRequirement) -> Self {
        self.requirements.push(requirement);
        self
    }
}

#[async_trait]
impl TransactionAction for ApplyUpdatesAction {
    async fn commit(self: Arc<Self>, _table: &Table) -> Result<ActionCommit> {
        Ok(ActionCommit::new(
            self.updates.clone(),
            self.requirements.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::spec::{
        NestedField, NullOrder, PrimitiveType, Schema, SortDirection, SortField, SortOrder,
        Transform, Type, UnboundPartitionSpec,
    };
    use crate::transaction::tests::make_v2_table;
    use crate::transaction::{ApplyUpdatesAction, Transaction, TransactionAction};
    use crate::{TableRequirement, TableUpdate};

    #[tokio::test]
    async fn test_apply_updates_evolves_schema_spec_and_sort_order() {
        let table = make_v2_table();
        let metadata = table.metadata();
        let current = metadata.current_schema();
        let renamed = current
            .as_struct()
            .fields()
            .iter()
            .map(|field| {
                if field.id == 1 {
                    Arc::new(NestedField::required(
                        1,
                        "x_renamed",
                        (*field.field_type).clone(),
                    ))
                } else {
                    Arc::clone(field)
                }
            })
            .chain([Arc::new(NestedField::optional(
                metadata.last_column_id() + 1,
                "added",
                Type::Primitive(PrimitiveType::String),
            ))])
            .collect::<Vec<_>>();
        let schema = Schema::builder()
            .with_schema_id(current.schema_id() + 1)
            .with_identifier_field_ids(current.identifier_field_ids())
            .with_fields(renamed)
            .build()
            .unwrap();
        let spec = UnboundPartitionSpec::builder()
            .add_partition_field(1, "x_bucket", Transform::Bucket(4))
            .unwrap()
            .build();
        let order = SortOrder::builder()
            .with_order_id(9)
            .with_sort_field(SortField {
                source_id: 1,
                transform: Transform::Identity,
                direction: SortDirection::Descending,
                null_order: NullOrder::Last,
            })
            .build_unbound()
            .unwrap();
        let action = ApplyUpdatesAction::new()
            .requirement(TableRequirement::CurrentSchemaIdMatch {
                current_schema_id: current.schema_id(),
            })
            .update(TableUpdate::AddSchema { schema })
            .update(TableUpdate::SetCurrentSchema { schema_id: -1 })
            .update(TableUpdate::AddSpec { spec })
            .update(TableUpdate::SetDefaultSpec { spec_id: -1 })
            .update(TableUpdate::AddSortOrder { sort_order: order })
            .update(TableUpdate::SetDefaultSortOrder { sort_order_id: -1 });

        let commit = Arc::new(action).commit(&table).await.unwrap();
        let (mut updates, mut requirements) = (Vec::new(), Vec::new());
        let evolved =
            Transaction::apply(table.clone(), commit, &mut updates, &mut requirements).unwrap();
        let evolved = evolved.metadata();

        assert_eq!(updates.len(), 6);
        assert_eq!(requirements.len(), 1);
        assert_eq!(
            evolved.current_schema().name_by_field_id(1),
            Some("x_renamed")
        );
        assert!(
            evolved
                .current_schema()
                .field_by_name("added")
                .unwrap()
                .required
                .eq(&false)
        );
        assert!(evolved.schema_by_id(current.schema_id()).is_some());
        assert_ne!(
            evolved.default_partition_spec_id(),
            metadata.default_partition_spec_id()
        );
        assert!(
            evolved
                .partition_spec_by_id(metadata.default_partition_spec_id())
                .is_some()
        );
        assert_ne!(
            evolved.default_sort_order_id(),
            metadata.default_sort_order_id()
        );
        assert!(
            evolved
                .sort_order_by_id(metadata.default_sort_order_id())
                .is_some()
        );
    }

    #[tokio::test]
    async fn test_apply_updates_refuses_a_stale_requirement() {
        let table = make_v2_table();
        let action =
            ApplyUpdatesAction::new().requirement(TableRequirement::CurrentSchemaIdMatch {
                current_schema_id: table.metadata().current_schema_id() + 7,
            });
        let commit = Arc::new(action).commit(&table).await.unwrap();
        assert!(
            Transaction::apply(table.clone(), commit, &mut Vec::new(), &mut Vec::new()).is_err()
        );
    }
}
