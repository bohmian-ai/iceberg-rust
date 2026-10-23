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

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use iceberg::encryption::kms::{KeyManagementClient, KmsClientFactory};
use iceberg::io::{FileIO, FileIOBuilder, StorageFactory};
use iceberg::spec::{TableMetadata, TableMetadataBuilder};
use iceberg::table::Table;
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, MetadataLocation, Namespace, NamespaceIdent, Result,
    Runtime, TableCommit, TableCreation, TableIdent, TableUpdate,
};
use sqlx::any::{AnyPoolOptions, AnyQueryResult, AnyRow, install_default_drivers};
use sqlx::{Any, AnyPool, AssertSqlSafe, Connection as _, Row, Transaction};

use crate::error::{
    from_sqlx_error, no_such_namespace_err, no_such_table_err, table_already_exists_err,
};

/// catalog URI
pub const SQL_CATALOG_PROP_URI: &str = "uri";
/// catalog warehouse location
pub const SQL_CATALOG_PROP_WAREHOUSE: &str = "warehouse";
/// catalog sql bind style
pub const SQL_CATALOG_PROP_BIND_STYLE: &str = "sql_bind_style";
/// Table property that carries each table's definition version.
///
/// When configured, creation stamps version `1`, a commit that sets the
/// property to `r + 1` is a definition edit from version `r`, and every
/// committed version keeps a retained reference to its metadata location
/// published in the same SQL transaction as the pointer swap.
pub const SQL_CATALOG_PROP_TABLE_VERSION: &str = "table-version-property";

static CATALOG_TABLE_NAME: &str = "iceberg_tables";
static CATALOG_FIELD_CATALOG_NAME: &str = "catalog_name";
static CATALOG_FIELD_TABLE_NAME: &str = "table_name";
static CATALOG_FIELD_TABLE_NAMESPACE: &str = "table_namespace";
static CATALOG_FIELD_METADATA_LOCATION_PROP: &str = "metadata_location";
static CATALOG_FIELD_PREVIOUS_METADATA_LOCATION_PROP: &str = "previous_metadata_location";
static CATALOG_FIELD_RECORD_TYPE: &str = "iceberg_type";
static CATALOG_FIELD_TABLE_RECORD_TYPE: &str = "TABLE";

static TABLE_VERSION_TABLE_NAME: &str = "iceberg_table_versions";
static TABLE_VERSION_FIELD_TABLE_UUID: &str = "table_uuid";
static TABLE_VERSION_FIELD_VERSION: &str = "table_version";

static NAMESPACE_TABLE_NAME: &str = "iceberg_namespace_properties";
static NAMESPACE_FIELD_NAME: &str = "namespace";
static NAMESPACE_FIELD_PROPERTY_KEY: &str = "property_key";
static NAMESPACE_FIELD_PROPERTY_VALUE: &str = "property_value";

static NAMESPACE_LOCATION_PROPERTY_KEY: &str = "location";

static MAX_CONNECTIONS: u32 = 10; // Default the SQL pool to 10 connections if not provided
static IDLE_TIMEOUT: u64 = 10; // Default the maximum idle timeout per connection to 10s before it is closed
static TEST_BEFORE_ACQUIRE: bool = true; // Default the health-check of each connection to enabled prior to returning

fn parse_pool_property<T>(
    props: &HashMap<String, String>,
    property: &'static str,
    default: T,
) -> Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    props.get(property).map_or(Ok(default), |value| {
        value.parse().map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to parse SQL catalog pool property",
            )
            .with_context("property", property)
            .with_context("value", value)
            .with_source(error)
        })
    })
}

/// Parses a canonical positive decimal table version.
///
/// Signs, leading zeros, whitespace, zero and overflow are refused.
fn parse_table_version(property: &str, value: &str) -> Result<i64> {
    value
        .parse::<i64>()
        .ok()
        .filter(|version| *version > 0 && version.to_string() == value)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("`{property}` must be a canonical positive decimal, not {value:?}"),
            )
        })
}

/// Builder for [`SqlCatalog`]
#[derive(Debug)]
pub struct SqlCatalogBuilder {
    config: SqlCatalogConfig,
    storage_factory: Option<Arc<dyn StorageFactory>>,
    kms_client_factory: Option<Arc<dyn KmsClientFactory>>,
    runtime: Option<Runtime>,
}

impl Default for SqlCatalogBuilder {
    fn default() -> Self {
        Self {
            config: SqlCatalogConfig {
                uri: "".to_string(),
                name: "".to_string(),
                warehouse_location: "".to_string(),
                sql_bind_style: SqlBindStyle::DollarNumeric,
                table_version_property: None,
                props: HashMap::new(),
            },
            storage_factory: None,
            kms_client_factory: None,
            runtime: None,
        }
    }
}

impl SqlCatalogBuilder {
    /// Configure the database URI
    ///
    /// If `SQL_CATALOG_PROP_URI` has a value set in `props` during `SqlCatalogBuilder::load`,
    /// that value takes precedence, and the value specified by this method will not be used.
    pub fn uri(mut self, uri: impl Into<String>) -> Self {
        self.config.uri = uri.into();
        self
    }

    /// Configure the warehouse location
    ///
    /// If `SQL_CATALOG_PROP_WAREHOUSE` has a value set in `props` during `SqlCatalogBuilder::load`,
    /// that value takes precedence, and the value specified by this method will not be used.
    pub fn warehouse_location(mut self, location: impl Into<String>) -> Self {
        self.config.warehouse_location = location.into();
        self
    }

    /// Configure the bound SQL Statement
    ///
    /// If `SQL_CATALOG_PROP_BIND_STYLE` has a value set in `props` during `SqlCatalogBuilder::load`,
    /// that value takes precedence, and the value specified by this method will not be used.
    pub fn sql_bind_style(mut self, sql_bind_style: SqlBindStyle) -> Self {
        self.config.sql_bind_style = sql_bind_style;
        self
    }

    /// Configure the any properties
    ///
    /// If the same key has values set in `props` during `SqlCatalogBuilder::load`,
    /// those values will take precedence.
    pub fn props(mut self, props: HashMap<String, String>) -> Self {
        for (k, v) in props {
            self.config.props.insert(k, v);
        }
        self
    }

    /// Set a new property on the property to be configured.
    /// When multiple methods are executed with the same key,
    /// the later-set value takes precedence.
    ///
    /// If the same key has values set in `props` during `SqlCatalogBuilder::load`,
    /// those values will take precedence.
    pub fn prop(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.props.insert(key.into(), value.into());
        self
    }
}

impl CatalogBuilder for SqlCatalogBuilder {
    type C = SqlCatalog;

    fn with_storage_factory(mut self, storage_factory: Arc<dyn StorageFactory>) -> Self {
        self.storage_factory = Some(storage_factory);
        self
    }

    fn with_kms_client_factory(mut self, kms_client_factory: Arc<dyn KmsClientFactory>) -> Self {
        self.kms_client_factory = Some(kms_client_factory);
        self
    }

    fn with_runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = Some(runtime);
        self
    }

    fn load(
        mut self,
        name: impl Into<String>,
        props: HashMap<String, String>,
    ) -> impl Future<Output = Result<Self::C>> + Send {
        for (k, v) in props {
            self.config.props.insert(k, v);
        }

        if let Some(uri) = self.config.props.remove(SQL_CATALOG_PROP_URI) {
            self.config.uri = uri;
        }
        if let Some(warehouse_location) = self.config.props.remove(SQL_CATALOG_PROP_WAREHOUSE) {
            self.config.warehouse_location = warehouse_location;
        }

        self.config.table_version_property =
            self.config.props.remove(SQL_CATALOG_PROP_TABLE_VERSION);

        let name = name.into();

        let mut valid_sql_bind_style = true;
        if let Some(sql_bind_style) = self.config.props.remove(SQL_CATALOG_PROP_BIND_STYLE) {
            if let Ok(sql_bind_style) = SqlBindStyle::from_str(&sql_bind_style) {
                self.config.sql_bind_style = sql_bind_style;
            } else {
                valid_sql_bind_style = false;
            }
        }

        let valid_name = !name.trim().is_empty();

        async move {
            if !valid_name {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Catalog name cannot be empty",
                ))
            } else if !valid_sql_bind_style {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "`{}` values are valid only if they're `{}` or `{}`",
                        SQL_CATALOG_PROP_BIND_STYLE,
                        SqlBindStyle::DollarNumeric,
                        SqlBindStyle::QMark
                    ),
                ))
            } else {
                self.config.name = name;
                let runtime = match self.runtime {
                    Some(rt) => rt,
                    None => Runtime::try_current()?,
                };
                let kms_client = match self.kms_client_factory {
                    Some(factory) => Some(factory.create_kms_client(&self.config.props).await?),
                    None => None,
                };
                SqlCatalog::new(self.config, self.storage_factory, runtime, kms_client).await
            }
        }
    }
}

/// A struct representing the SQL catalog configuration.
///
/// This struct contains various parameters that are used to configure a SQL catalog,
/// such as the database URI, warehouse location, and file I/O settings.
/// You are required to provide a `SqlBindStyle`, which determines how SQL statements will be bound to values in the catalog.
/// The options available for this parameter include:
/// - `SqlBindStyle::DollarNumeric`: Binds SQL statements using `$1`, `$2`, etc., as placeholders. This is for PostgreSQL databases.
/// - `SqlBindStyle::QuestionMark`: Binds SQL statements using `?` as a placeholder. This is for MySQL and SQLite databases.
#[derive(Debug)]
struct SqlCatalogConfig {
    uri: String,
    name: String,
    warehouse_location: String,
    sql_bind_style: SqlBindStyle,
    table_version_property: Option<String>,
    props: HashMap<String, String>,
}

#[derive(Debug)]
/// Sql catalog implementation.
pub struct SqlCatalog {
    name: String,
    connection: AnyPool,
    warehouse_location: String,
    fileio: FileIO,
    sql_bind_style: SqlBindStyle,
    table_version_property: Option<String>,
    runtime: Runtime,
    kms_client: Option<Arc<dyn KeyManagementClient>>,
}

#[derive(Debug, PartialEq, strum::EnumString, strum::Display)]
/// Set the SQL parameter bind style to either $1..$N (Postgres style) or ? (SQLite/MySQL/MariaDB)
pub enum SqlBindStyle {
    /// DollarNumeric uses parameters of the form `$1..$N``, which is the Postgres style
    DollarNumeric,
    /// QMark uses parameters of the form `?` which is the style for other dialects (SQLite/MySQL/MariaDB)
    QMark,
}

impl SqlCatalog {
    /// Create new sql catalog instance
    async fn new(
        config: SqlCatalogConfig,
        storage_factory: Option<Arc<dyn StorageFactory>>,
        runtime: Runtime,
        kms_client: Option<Arc<dyn KeyManagementClient>>,
    ) -> Result<Self> {
        let factory = storage_factory.ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "StorageFactory must be provided for SqlCatalog. Use `with_storage_factory` to configure it.",
            )
        })?;
        // Forward catalog props so storage-backend keys reach the FileIO.
        // Unrecognized keys are ignored by backends.
        let fileio = FileIOBuilder::new(factory)
            .with_props(config.props.clone())
            .build();

        install_default_drivers();
        let max_connections =
            parse_pool_property(&config.props, "pool.max-connections", MAX_CONNECTIONS)?;
        let idle_timeout = parse_pool_property(&config.props, "pool.idle-timeout", IDLE_TIMEOUT)?;
        let test_before_acquire = parse_pool_property(
            &config.props,
            "pool.test-before-acquire",
            TEST_BEFORE_ACQUIRE,
        )?;

        let test_idle_after = config
            .props
            .contains_key("pool.test-idle-after-ms")
            .then(|| parse_pool_property(&config.props, "pool.test-idle-after-ms", 0_u64))
            .transpose()?
            .map(Duration::from_millis);

        let options = AnyPoolOptions::new()
            .max_connections(max_connections)
            .idle_timeout(Duration::from_secs(idle_timeout));
        // A connection used moments ago is overwhelmingly still alive, so an
        // idle threshold pings only connections that sat long enough to have
        // been dropped, instead of paying a round trip on every checkout.
        let options = match test_idle_after {
            Some(threshold) if test_before_acquire => options
                .test_before_acquire(false)
                .before_acquire(move |conn, meta| {
                    Box::pin(async move {
                        if meta.idle_for > threshold {
                            conn.ping().await?;
                        }
                        Ok(true)
                    })
                }),
            _ => options.test_before_acquire(test_before_acquire),
        };
        let pool = options
            .connect(&config.uri)
            .await
            .map_err(from_sqlx_error)?;

        sqlx::query(AssertSqlSafe(format!(
            "CREATE TABLE IF NOT EXISTS {CATALOG_TABLE_NAME} (
                {CATALOG_FIELD_CATALOG_NAME} VARCHAR(255) NOT NULL,
                {CATALOG_FIELD_TABLE_NAMESPACE} VARCHAR(255) NOT NULL,
                {CATALOG_FIELD_TABLE_NAME} VARCHAR(255) NOT NULL,
                {CATALOG_FIELD_METADATA_LOCATION_PROP} VARCHAR(1000),
                {CATALOG_FIELD_PREVIOUS_METADATA_LOCATION_PROP} VARCHAR(1000),
                {CATALOG_FIELD_RECORD_TYPE} VARCHAR(5),
                PRIMARY KEY ({CATALOG_FIELD_CATALOG_NAME}, {CATALOG_FIELD_TABLE_NAMESPACE}, {CATALOG_FIELD_TABLE_NAME}))"
        )))
        .execute(&pool)
        .await
        .map_err(from_sqlx_error)?;

        sqlx::query(AssertSqlSafe(format!(
            "CREATE TABLE IF NOT EXISTS {NAMESPACE_TABLE_NAME} (
                {CATALOG_FIELD_CATALOG_NAME} VARCHAR(255) NOT NULL,
                {NAMESPACE_FIELD_NAME} VARCHAR(255) NOT NULL,
                {NAMESPACE_FIELD_PROPERTY_KEY} VARCHAR(255),
                {NAMESPACE_FIELD_PROPERTY_VALUE} VARCHAR(1000),
                PRIMARY KEY ({CATALOG_FIELD_CATALOG_NAME}, {NAMESPACE_FIELD_NAME}, {NAMESPACE_FIELD_PROPERTY_KEY}))"
        )))
        .execute(&pool)
        .await
        .map_err(from_sqlx_error)?;

        sqlx::query(AssertSqlSafe(format!(
            "CREATE TABLE IF NOT EXISTS {TABLE_VERSION_TABLE_NAME} (
                {CATALOG_FIELD_CATALOG_NAME} VARCHAR(255) NOT NULL,
                {CATALOG_FIELD_TABLE_NAMESPACE} VARCHAR(255) NOT NULL,
                {CATALOG_FIELD_TABLE_NAME} VARCHAR(255) NOT NULL,
                {TABLE_VERSION_FIELD_TABLE_UUID} VARCHAR(36) NOT NULL,
                {TABLE_VERSION_FIELD_VERSION} BIGINT NOT NULL CHECK ({TABLE_VERSION_FIELD_VERSION} > 0),
                {CATALOG_FIELD_METADATA_LOCATION_PROP} VARCHAR(1000) NOT NULL,
                PRIMARY KEY ({CATALOG_FIELD_CATALOG_NAME}, {TABLE_VERSION_FIELD_TABLE_UUID}, {TABLE_VERSION_FIELD_VERSION}))"
        )))
        .execute(&pool)
        .await
        .map_err(from_sqlx_error)?;

        Ok(SqlCatalog {
            name: config.name.to_owned(),
            connection: pool,
            warehouse_location: config.warehouse_location,
            fileio,
            sql_bind_style: config.sql_bind_style,
            table_version_property: config.table_version_property,
            runtime,
            kms_client,
        })
    }

    /// Reads only the table's current metadata pointer from the catalog row.
    ///
    /// Every commit writes a new metadata file at a fresh
    /// `<version>-<uuid>.metadata.json` location and swaps this pointer, so an
    /// unchanged pointer names an unchanged metadata document. Callers that
    /// already hold that document can recheck the pointer without re-reading
    /// it. Returns `None` when the table does not exist.
    pub async fn load_metadata_location(&self, identifier: &TableIdent) -> Result<Option<String>> {
        let rows = self
            .fetch_rows(
                &format!(
                    "SELECT {CATALOG_FIELD_METADATA_LOCATION_PROP}
                     FROM {CATALOG_TABLE_NAME}
                     WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                      AND {CATALOG_FIELD_TABLE_NAME} = ?
                      AND {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                      AND (
                        {CATALOG_FIELD_RECORD_TYPE} = '{CATALOG_FIELD_TABLE_RECORD_TYPE}'
                        OR {CATALOG_FIELD_RECORD_TYPE} IS NULL
                      )"
                ),
                vec![
                    Some(&self.name),
                    Some(identifier.name()),
                    Some(&identifier.namespace().join(".")),
                ],
            )
            .await?;
        rows.first()
            .map(|row| {
                row.try_get::<String, _>(CATALOG_FIELD_METADATA_LOCATION_PROP)
                    .map_err(from_sqlx_error)
            })
            .transpose()
    }

    /// Reads the retained metadata location of one committed table version.
    ///
    /// `table_uuid` is the table's hyphenated metadata UUID, so a recreated
    /// table under the same name never resolves its predecessor's versions.
    /// Returns `None` when that version was never committed for that table.
    pub async fn load_table_version_location(
        &self,
        identifier: &TableIdent,
        table_uuid: &str,
        table_version: i64,
    ) -> Result<Option<String>> {
        let table_version = table_version.to_string();
        let rows = self
            .fetch_rows(
                &format!(
                    "SELECT {CATALOG_FIELD_METADATA_LOCATION_PROP}
                     FROM {TABLE_VERSION_TABLE_NAME}
                     WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                      AND {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                      AND {CATALOG_FIELD_TABLE_NAME} = ?
                      AND {TABLE_VERSION_FIELD_TABLE_UUID} = ?
                      AND {TABLE_VERSION_FIELD_VERSION} = CAST(? AS BIGINT)"
                ),
                vec![
                    Some(&self.name),
                    Some(&identifier.namespace().join(".")),
                    Some(identifier.name()),
                    Some(table_uuid),
                    Some(&table_version),
                ],
            )
            .await?;
        rows.first()
            .map(|row| {
                row.try_get::<String, _>(CATALOG_FIELD_METADATA_LOCATION_PROP)
                    .map_err(from_sqlx_error)
            })
            .transpose()
    }

    /// Reads the configured table version of `metadata`.
    ///
    /// Returns `None` when no version property is configured. A configured
    /// property that is missing or not a canonical positive decimal is
    /// refused, never defaulted.
    fn table_version(&self, metadata: &TableMetadata) -> Result<Option<i64>> {
        let Some(property) = &self.table_version_property else {
            return Ok(None);
        };
        let value = metadata.properties().get(property).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("table metadata has no `{property}`"),
            )
        })?;
        parse_table_version(property, value).map(Some)
    }

    /// Returns the version a commit's updates publish, if it edits the
    /// definition.
    ///
    /// A commit that sets the configured property is a definition edit and
    /// must carry at least one other update. Removing the property, or
    /// changing the current schema, default spec or default sort order without
    /// setting it, is refused.
    fn requested_table_version(&self, updates: &[TableUpdate]) -> Result<Option<i64>> {
        let Some(property) = &self.table_version_property else {
            return Ok(None);
        };
        let mut requested = None;
        let mut other_updates = 0_usize;
        let mut definition_updates = 0_usize;
        for update in updates {
            match update {
                TableUpdate::SetProperties { updates } => {
                    if let Some(value) = updates.get(property) {
                        requested = Some(parse_table_version(property, value)?);
                    }
                    other_updates += updates.keys().filter(|key| *key != property).count();
                }
                TableUpdate::RemoveProperties { removals } => {
                    if removals.contains(property) {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!("`{property}` cannot be removed"),
                        ));
                    }
                    other_updates += removals.len();
                }
                TableUpdate::SetCurrentSchema { .. }
                | TableUpdate::SetDefaultSpec { .. }
                | TableUpdate::SetDefaultSortOrder { .. } => {
                    definition_updates += 1;
                    other_updates += 1;
                }
                _ => other_updates += 1,
            }
        }
        if requested.is_none() && definition_updates > 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("a definition edit must advance `{property}`"),
            ));
        }
        if requested.is_some() && other_updates == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("a `{property}` change must accompany a definition edit"),
            ));
        }
        Ok(requested)
    }

    /// Returns whether `after` declares a different effective definition
    /// than `before`.
    ///
    /// The definition is the current schema, default partition spec, default
    /// sort order and every property except the configured table version;
    /// snapshot, log and timestamp bookkeeping is not part of it. A
    /// version-intended edit that leaves all of them unchanged would consume
    /// a version and stale every writer bound to the current one for nothing.
    fn edits_definition(&self, before: &TableMetadata, after: &TableMetadata) -> bool {
        let version = self.table_version_property.as_ref();
        let properties = |metadata: &TableMetadata| {
            metadata
                .properties()
                .iter()
                .filter(|(key, _)| Some(*key) != version)
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<HashMap<_, _>>()
        };
        before.current_schema() != after.current_schema()
            || before.default_partition_spec() != after.default_partition_spec()
            || before.default_sort_order() != after.default_sort_order()
            || properties(before) != properties(after)
    }

    /// Stages the retained reference of one committed table version.
    async fn insert_table_version(
        &self,
        identifier: &TableIdent,
        metadata: &TableMetadata,
        table_version: i64,
        metadata_location: &str,
        transaction: &mut Transaction<'_, Any>,
    ) -> Result<()> {
        let table_uuid = metadata.uuid().to_string();
        let table_version = table_version.to_string();
        self.execute(
            &format!(
                "INSERT INTO {TABLE_VERSION_TABLE_NAME}
                 ({CATALOG_FIELD_CATALOG_NAME}, {CATALOG_FIELD_TABLE_NAMESPACE}, {CATALOG_FIELD_TABLE_NAME}, {TABLE_VERSION_FIELD_TABLE_UUID}, {TABLE_VERSION_FIELD_VERSION}, {CATALOG_FIELD_METADATA_LOCATION_PROP})
                 VALUES (?, ?, ?, ?, CAST(? AS BIGINT), ?)"
            ),
            vec![
                Some(&self.name),
                Some(&identifier.namespace().join(".")),
                Some(identifier.name()),
                Some(&table_uuid),
                Some(&table_version),
                Some(metadata_location),
            ],
            Some(transaction),
        )
        .await?;
        Ok(())
    }

    /// Commits one publication transaction.
    ///
    /// An uncertain COMMIT is resolved from the authoritative pointer: the
    /// publication succeeded exactly when the pointer names `published`, and
    /// the pointer and its version reference commit together.
    async fn commit_publication(
        &self,
        transaction: Transaction<'_, Any>,
        identifier: &TableIdent,
        published: &str,
    ) -> Result<()> {
        let Err(error) = transaction.commit().await else {
            return Ok(());
        };
        match self.load_metadata_location(identifier).await {
            Ok(Some(location)) if location == published => Ok(()),
            _ => Err(from_sqlx_error(error)),
        }
    }

    /// SQLX Any does not implement PostgresSQL bindings, so we have to do this.
    fn replace_placeholders(&self, query: &str) -> String {
        match self.sql_bind_style {
            SqlBindStyle::DollarNumeric => {
                let mut count = 1;
                query
                    .chars()
                    .fold(String::with_capacity(query.len()), |mut acc, c| {
                        if c == '?' {
                            acc.push('$');
                            acc.push_str(&count.to_string());
                            count += 1;
                        } else {
                            acc.push(c);
                        }
                        acc
                    })
            }
            _ => query.to_owned(),
        }
    }

    /// Fetch a vec of AnyRows from a given query
    async fn fetch_rows(&self, query: &str, args: Vec<Option<&str>>) -> Result<Vec<AnyRow>> {
        let query_with_placeholders = self.replace_placeholders(query);

        let mut sqlx_query = sqlx::query(AssertSqlSafe(query_with_placeholders));
        for arg in args {
            sqlx_query = sqlx_query.bind(arg);
        }

        sqlx_query
            .fetch_all(&self.connection)
            .await
            .map_err(from_sqlx_error)
    }

    /// Execute statements in a transaction, provided or not
    async fn execute(
        &self,
        query: &str,
        args: Vec<Option<&str>>,
        transaction: Option<&mut Transaction<'_, Any>>,
    ) -> Result<AnyQueryResult> {
        let query_with_placeholders = self.replace_placeholders(query);

        let mut sqlx_query = sqlx::query(AssertSqlSafe(query_with_placeholders));
        for arg in args {
            sqlx_query = sqlx_query.bind(arg);
        }

        match transaction {
            Some(t) => sqlx_query.execute(&mut **t).await.map_err(from_sqlx_error),
            None => {
                let mut tx = self.connection.begin().await.map_err(from_sqlx_error)?;
                let result = sqlx_query
                    .execute(&mut *tx)
                    .await
                    .map_err(from_sqlx_error)?;
                tx.commit().await.map_err(from_sqlx_error)?;
                Ok(result)
            }
        }
    }
}

#[async_trait]
impl Catalog for SqlCatalog {
    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        // UNION will remove duplicates.
        let all_namespaces_stmt = format!(
            "SELECT {CATALOG_FIELD_TABLE_NAMESPACE}
             FROM {CATALOG_TABLE_NAME}
             WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
             UNION
             SELECT {NAMESPACE_FIELD_NAME}
             FROM {NAMESPACE_TABLE_NAME}
             WHERE {CATALOG_FIELD_CATALOG_NAME} = ?"
        );

        let namespace_rows = self
            .fetch_rows(
                &all_namespaces_stmt,
                vec![Some(&self.name), Some(&self.name)],
            )
            .await?;

        let mut namespaces = HashSet::<NamespaceIdent>::with_capacity(namespace_rows.len());

        if let Some(parent) = parent {
            if self.namespace_exists(parent).await? {
                let parent_str = parent.join(".");

                for row in namespace_rows.iter() {
                    let nsp = row.try_get::<String, _>(0).map_err(from_sqlx_error)?;
                    // if parent = a, then we only want to see a.b, a.c returned.
                    if nsp != parent_str && nsp.starts_with(&parent_str) {
                        namespaces.insert(NamespaceIdent::from_strs(nsp.split("."))?);
                    }
                }

                Ok(namespaces.into_iter().collect::<Vec<NamespaceIdent>>())
            } else {
                no_such_namespace_err(parent)
            }
        } else {
            for row in namespace_rows.iter() {
                let nsp = row.try_get::<String, _>(0).map_err(from_sqlx_error)?;
                let mut levels = nsp.split(".").collect::<Vec<&str>>();
                if !levels.is_empty() {
                    let first_level = levels.drain(..1).collect::<Vec<&str>>();
                    namespaces.insert(NamespaceIdent::from_strs(first_level)?);
                }
            }

            Ok(namespaces.into_iter().collect::<Vec<NamespaceIdent>>())
        }
    }

    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        let exists = self.namespace_exists(namespace).await?;

        if exists {
            return Err(Error::new(
                ErrorKind::NamespaceAlreadyExists,
                format!("Namespace {namespace:?} already exists"),
            ));
        }

        let namespace_str = namespace.join(".");
        let insert = format!(
            "INSERT INTO {NAMESPACE_TABLE_NAME} ({CATALOG_FIELD_CATALOG_NAME}, {NAMESPACE_FIELD_NAME}, {NAMESPACE_FIELD_PROPERTY_KEY}, {NAMESPACE_FIELD_PROPERTY_VALUE})
             VALUES (?, ?, ?, ?)");
        if !properties.is_empty() {
            let mut insert_properties = properties.clone();
            insert_properties.insert("exists".to_string(), "true".to_string());

            let mut query_args = Vec::with_capacity(insert_properties.len() * 4);
            let mut insert_stmt = insert.clone();
            for (index, (key, value)) in insert_properties.iter().enumerate() {
                query_args.extend_from_slice(&[
                    Some(self.name.as_str()),
                    Some(namespace_str.as_str()),
                    Some(key.as_str()),
                    Some(value.as_str()),
                ]);
                if index > 0 {
                    insert_stmt.push_str(", (?, ?, ?, ?)");
                }
            }

            self.execute(&insert_stmt, query_args, None).await?;

            Ok(Namespace::with_properties(
                namespace.clone(),
                insert_properties,
            ))
        } else {
            // set a default property of exists = true
            self.execute(
                &insert,
                vec![
                    Some(&self.name),
                    Some(&namespace_str),
                    Some("exists"),
                    Some("true"),
                ],
                None,
            )
            .await?;
            Ok(Namespace::with_properties(namespace.clone(), properties))
        }
    }

    async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
        let exists = self.namespace_exists(namespace).await?;
        if exists {
            let namespace_props = self
                .fetch_rows(
                    &format!(
                        "SELECT
                            {NAMESPACE_FIELD_NAME},
                            {NAMESPACE_FIELD_PROPERTY_KEY},
                            {NAMESPACE_FIELD_PROPERTY_VALUE}
                            FROM {NAMESPACE_TABLE_NAME}
                            WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                            AND {NAMESPACE_FIELD_NAME} = ?"
                    ),
                    vec![Some(&self.name), Some(&namespace.join("."))],
                )
                .await?;

            let mut properties = HashMap::with_capacity(namespace_props.len());

            for row in namespace_props {
                let key = row
                    .try_get::<String, _>(NAMESPACE_FIELD_PROPERTY_KEY)
                    .map_err(from_sqlx_error)?;
                let value = row
                    .try_get::<String, _>(NAMESPACE_FIELD_PROPERTY_VALUE)
                    .map_err(from_sqlx_error)?;

                properties.insert(key, value);
            }

            Ok(Namespace::with_properties(namespace.clone(), properties))
        } else {
            no_such_namespace_err(namespace)
        }
    }

    async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
        let namespace_str = namespace.join(".");

        let table_namespaces = self
            .fetch_rows(
                &format!(
                    "SELECT 1 FROM {CATALOG_TABLE_NAME}
                     WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                      AND {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                     LIMIT 1"
                ),
                vec![Some(&self.name), Some(&namespace_str)],
            )
            .await?;

        if !table_namespaces.is_empty() {
            Ok(true)
        } else {
            let namespaces = self
                .fetch_rows(
                    &format!(
                        "SELECT 1 FROM {NAMESPACE_TABLE_NAME}
                         WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                          AND {NAMESPACE_FIELD_NAME} = ?
                         LIMIT 1"
                    ),
                    vec![Some(&self.name), Some(&namespace_str)],
                )
                .await?;
            if !namespaces.is_empty() {
                Ok(true)
            } else {
                Ok(false)
            }
        }
    }

    async fn update_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<()> {
        let exists = self.namespace_exists(namespace).await?;
        if exists {
            let existing_properties = self.get_namespace(namespace).await?.properties().clone();
            let namespace_str = namespace.join(".");

            let mut updates = vec![];
            let mut inserts = vec![];

            for (key, value) in properties.iter() {
                if existing_properties.contains_key(key) {
                    if existing_properties.get(key) != Some(value) {
                        updates.push((key, value));
                    }
                } else {
                    inserts.push((key, value));
                }
            }

            let mut tx = self.connection.begin().await.map_err(from_sqlx_error)?;
            let update_stmt = format!(
                "UPDATE {NAMESPACE_TABLE_NAME} SET {NAMESPACE_FIELD_PROPERTY_VALUE} = ?
                 WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                 AND {NAMESPACE_FIELD_NAME} = ?
                 AND {NAMESPACE_FIELD_PROPERTY_KEY} = ?"
            );

            let insert_stmt = format!(
                "INSERT INTO {NAMESPACE_TABLE_NAME} ({CATALOG_FIELD_CATALOG_NAME}, {NAMESPACE_FIELD_NAME}, {NAMESPACE_FIELD_PROPERTY_KEY}, {NAMESPACE_FIELD_PROPERTY_VALUE})
                 VALUES (?, ?, ?, ?)"
            );

            for (key, value) in updates {
                self.execute(
                    &update_stmt,
                    vec![
                        Some(value),
                        Some(&self.name),
                        Some(&namespace_str),
                        Some(key),
                    ],
                    Some(&mut tx),
                )
                .await?;
            }

            for (key, value) in inserts {
                self.execute(
                    &insert_stmt,
                    vec![
                        Some(&self.name),
                        Some(&namespace_str),
                        Some(key),
                        Some(value),
                    ],
                    Some(&mut tx),
                )
                .await?;
            }

            let _ = tx.commit().await.map_err(from_sqlx_error)?;

            Ok(())
        } else {
            no_such_namespace_err(namespace)
        }
    }

    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
        let exists = self.namespace_exists(namespace).await?;
        if exists {
            // if there are tables in the namespace, don't allow drop.
            let tables = self.list_tables(namespace).await?;
            if !tables.is_empty() {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "Namespace {:?} is not empty. {} tables exist.",
                        namespace,
                        tables.len()
                    ),
                ));
            }

            self.execute(
                &format!(
                    "DELETE FROM {NAMESPACE_TABLE_NAME}
                     WHERE {NAMESPACE_FIELD_NAME} = ?
                      AND {CATALOG_FIELD_CATALOG_NAME} = ?"
                ),
                vec![Some(&namespace.join(".")), Some(&self.name)],
                None,
            )
            .await?;

            Ok(())
        } else {
            no_such_namespace_err(namespace)
        }
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        let exists = self.namespace_exists(namespace).await?;
        if exists {
            let rows = self
                .fetch_rows(
                    &format!(
                        "SELECT {CATALOG_FIELD_TABLE_NAME},
                                {CATALOG_FIELD_TABLE_NAMESPACE}
                         FROM {CATALOG_TABLE_NAME}
                         WHERE {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                          AND {CATALOG_FIELD_CATALOG_NAME} = ?
                          AND (
                                {CATALOG_FIELD_RECORD_TYPE} = '{CATALOG_FIELD_TABLE_RECORD_TYPE}'
                                OR {CATALOG_FIELD_RECORD_TYPE} IS NULL
                          )",
                    ),
                    vec![Some(&namespace.join(".")), Some(&self.name)],
                )
                .await?;

            let mut tables = HashSet::<TableIdent>::with_capacity(rows.len());

            for row in rows.iter() {
                let tbl = row
                    .try_get::<String, _>(CATALOG_FIELD_TABLE_NAME)
                    .map_err(from_sqlx_error)?;
                let ns_strs = row
                    .try_get::<String, _>(CATALOG_FIELD_TABLE_NAMESPACE)
                    .map_err(from_sqlx_error)?;
                let ns = NamespaceIdent::from_strs(ns_strs.split("."))?;
                tables.insert(TableIdent::new(ns, tbl));
            }

            Ok(tables.into_iter().collect::<Vec<TableIdent>>())
        } else {
            no_such_namespace_err(namespace)
        }
    }

    async fn table_exists(&self, identifier: &TableIdent) -> Result<bool> {
        let namespace = identifier.namespace().join(".");
        let table_name = identifier.name();
        let table_counts = self
            .fetch_rows(
                &format!(
                    "SELECT 1
                     FROM {CATALOG_TABLE_NAME}
                     WHERE {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                      AND {CATALOG_FIELD_CATALOG_NAME} = ?
                      AND {CATALOG_FIELD_TABLE_NAME} = ?
                      AND (
                        {CATALOG_FIELD_RECORD_TYPE} = '{CATALOG_FIELD_TABLE_RECORD_TYPE}'
                        OR {CATALOG_FIELD_RECORD_TYPE} IS NULL
                      )"
                ),
                vec![Some(&namespace), Some(&self.name), Some(table_name)],
            )
            .await?;

        if !table_counts.is_empty() {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn drop_table(&self, identifier: &TableIdent) -> Result<()> {
        if !self.table_exists(identifier).await? {
            return no_such_table_err(identifier);
        }

        self.execute(
            &format!(
                "DELETE FROM {CATALOG_TABLE_NAME}
                 WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                  AND {CATALOG_FIELD_TABLE_NAME} = ?
                  AND {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                  AND (
                    {CATALOG_FIELD_RECORD_TYPE} = '{CATALOG_FIELD_TABLE_RECORD_TYPE}'
                    OR {CATALOG_FIELD_RECORD_TYPE} IS NULL
                  )"
            ),
            vec![
                Some(&self.name),
                Some(identifier.name()),
                Some(&identifier.namespace().join(".")),
            ],
            None,
        )
        .await?;

        Ok(())
    }

    async fn purge_table(&self, table: &TableIdent) -> Result<()> {
        let table_info = self.load_table(table).await?;
        self.drop_table(table).await?;
        iceberg::drop_table_data(&table_info).await
    }

    async fn load_table(&self, identifier: &TableIdent) -> Result<Table> {
        if !self.table_exists(identifier).await? {
            return no_such_table_err(identifier);
        }

        let Some(tbl_metadata_location) = self.load_metadata_location(identifier).await? else {
            return no_such_table_err(identifier);
        };

        let metadata = TableMetadata::read_from(&self.fileio, &tbl_metadata_location).await?;

        let mut builder = Table::builder()
            .file_io(self.fileio.clone())
            .identifier(identifier.clone())
            .metadata_location(tbl_metadata_location)
            .metadata(metadata)
            .runtime(self.runtime.clone());
        if let Some(kms_client) = self.kms_client.clone() {
            builder = builder.kms_client(kms_client);
        }
        Ok(builder.build()?)
    }

    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        creation: TableCreation,
    ) -> Result<Table> {
        if !self.namespace_exists(namespace).await? {
            return no_such_namespace_err(namespace);
        }

        let tbl_name = creation.name.clone();
        let tbl_ident = TableIdent::new(namespace.clone(), tbl_name.clone());

        if self.table_exists(&tbl_ident).await? {
            return table_already_exists_err(&tbl_ident);
        }

        let tbl_creation = if creation.location.is_some() {
            creation
        } else {
            // fall back to namespace-specific location
            // and then to warehouse location
            let nsp_properties = self.get_namespace(namespace).await?.properties().clone();
            let nsp_location = match nsp_properties.get(NAMESPACE_LOCATION_PROPERTY_KEY) {
                Some(location) => location.clone(),
                None => {
                    format!(
                        "{}/{}",
                        self.warehouse_location.clone(),
                        namespace.join("/")
                    )
                }
            };

            let tbl_location = format!("{}/{}", nsp_location, tbl_ident.name());

            TableCreation {
                location: Some(tbl_location),
                ..creation
            }
        };

        let tbl_creation = match &self.table_version_property {
            Some(property) => {
                let mut tbl_creation = tbl_creation;
                let version = tbl_creation
                    .properties
                    .entry(property.clone())
                    .or_insert_with(|| "1".to_string());
                if version != "1" {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("a new table starts at `{property}` 1, not {version}"),
                    ));
                }
                tbl_creation
            }
            None => tbl_creation,
        };

        let tbl_metadata = TableMetadataBuilder::from_table_creation(tbl_creation)?
            .build()?
            .metadata;
        let tbl_metadata_location = MetadataLocation::try_new_with_metadata(&tbl_metadata)?;

        tbl_metadata
            .write_to(&self.fileio, &tbl_metadata_location)
            .await?;

        let tbl_metadata_location_str = tbl_metadata_location.to_string();
        let mut tx = self.connection.begin().await.map_err(from_sqlx_error)?;
        self.execute(&format!(
            "INSERT INTO {CATALOG_TABLE_NAME}
             ({CATALOG_FIELD_CATALOG_NAME}, {CATALOG_FIELD_TABLE_NAMESPACE}, {CATALOG_FIELD_TABLE_NAME}, {CATALOG_FIELD_METADATA_LOCATION_PROP}, {CATALOG_FIELD_RECORD_TYPE})
             VALUES (?, ?, ?, ?, ?)
            "), vec![Some(&self.name), Some(&namespace.join(".")), Some(&tbl_name.clone()), Some(&tbl_metadata_location_str), Some(CATALOG_FIELD_TABLE_RECORD_TYPE)], Some(&mut tx)).await?;
        if let Some(version) = self.table_version(&tbl_metadata)? {
            self.insert_table_version(
                &tbl_ident,
                &tbl_metadata,
                version,
                &tbl_metadata_location_str,
                &mut tx,
            )
            .await?;
        }
        self.commit_publication(tx, &tbl_ident, &tbl_metadata_location_str)
            .await?;

        let mut builder = Table::builder()
            .file_io(self.fileio.clone())
            .metadata_location(tbl_metadata_location_str)
            .identifier(tbl_ident)
            .metadata(tbl_metadata)
            .runtime(self.runtime.clone());
        if let Some(kms_client) = self.kms_client.clone() {
            builder = builder.kms_client(kms_client);
        }
        Ok(builder.build()?)
    }

    async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
        if src == dest {
            return Ok(());
        }

        if !self.table_exists(src).await? {
            return no_such_table_err(src);
        }

        if !self.namespace_exists(dest.namespace()).await? {
            return no_such_namespace_err(dest.namespace());
        }

        if self.table_exists(dest).await? {
            return table_already_exists_err(dest);
        }

        self.execute(
            &format!(
                "UPDATE {CATALOG_TABLE_NAME}
                 SET {CATALOG_FIELD_TABLE_NAME} = ?, {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                 WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                  AND {CATALOG_FIELD_TABLE_NAME} = ?
                  AND {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                  AND (
                    {CATALOG_FIELD_RECORD_TYPE} = '{CATALOG_FIELD_TABLE_RECORD_TYPE}'
                    OR {CATALOG_FIELD_RECORD_TYPE} IS NULL
                )"
            ),
            vec![
                Some(dest.name()),
                Some(&dest.namespace().join(".")),
                Some(&self.name),
                Some(src.name()),
                Some(&src.namespace().join(".")),
            ],
            None,
        )
        .await?;

        Ok(())
    }

    async fn register_table(
        &self,
        table_ident: &TableIdent,
        metadata_location: String,
    ) -> Result<Table> {
        if self.table_exists(table_ident).await? {
            return table_already_exists_err(table_ident);
        }

        let metadata = TableMetadata::read_from(&self.fileio, &metadata_location).await?;

        let namespace = table_ident.namespace();
        let tbl_name = table_ident.name().to_string();

        let mut tx = self.connection.begin().await.map_err(from_sqlx_error)?;
        self.execute(&format!(
            "INSERT INTO {CATALOG_TABLE_NAME}
             ({CATALOG_FIELD_CATALOG_NAME}, {CATALOG_FIELD_TABLE_NAMESPACE}, {CATALOG_FIELD_TABLE_NAME}, {CATALOG_FIELD_METADATA_LOCATION_PROP}, {CATALOG_FIELD_RECORD_TYPE})
             VALUES (?, ?, ?, ?, ?)
            "), vec![Some(&self.name), Some(&namespace.join(".")), Some(&tbl_name), Some(&metadata_location), Some(CATALOG_FIELD_TABLE_RECORD_TYPE)], Some(&mut tx)).await?;
        if let Some(version) = self.table_version(&metadata)? {
            self.insert_table_version(table_ident, &metadata, version, &metadata_location, &mut tx)
                .await?;
        }
        self.commit_publication(tx, table_ident, &metadata_location)
            .await?;

        let mut builder = Table::builder()
            .identifier(table_ident.clone())
            .metadata_location(metadata_location)
            .metadata(metadata)
            .file_io(self.fileio.clone())
            .runtime(self.runtime.clone());
        if let Some(kms_client) = self.kms_client.clone() {
            builder = builder.kms_client(kms_client);
        }
        Ok(builder.build()?)
    }

    /// Updates an existing table within the SQL catalog.
    ///
    /// With a configured table-version property, a commit that sets it to
    /// `r + 1` must apply to a base at version `r`; any other base is a
    /// non-retryable stale-version conflict. The pointer swap and the new
    /// version's retained reference commit in one SQL transaction.
    async fn list_table_version_locations(
        &self,
        table: &TableIdent,
        table_uuid: &str,
    ) -> Result<Vec<String>> {
        let rows = self
            .fetch_rows(
                &format!(
                    "SELECT {CATALOG_FIELD_METADATA_LOCATION_PROP}
                     FROM {TABLE_VERSION_TABLE_NAME}
                     WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                      AND {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                      AND {CATALOG_FIELD_TABLE_NAME} = ?
                      AND {TABLE_VERSION_FIELD_TABLE_UUID} = ?
                     ORDER BY {TABLE_VERSION_FIELD_VERSION}"
                ),
                vec![
                    Some(&self.name),
                    Some(&table.namespace().join(".")),
                    Some(table.name()),
                    Some(table_uuid),
                ],
            )
            .await?;
        rows.iter()
            .map(|row| {
                row.try_get::<String, _>(CATALOG_FIELD_METADATA_LOCATION_PROP)
                    .map_err(from_sqlx_error)
            })
            .collect()
    }

    async fn update_table(&self, commit: TableCommit) -> Result<Table> {
        let table_ident = commit.identifier().clone();
        let requested_version = self.requested_table_version(commit.updates())?;
        let current_table = self.load_table(&table_ident).await?;
        let current_metadata_location = current_table.metadata_location_result()?.to_string();
        if let Some(requested) = requested_version {
            let base = self.table_version(current_table.metadata())?;
            if base.and_then(|base| base.checked_add(1)) != Some(requested) {
                return Err(Error::new(
                    ErrorKind::CatalogCommitConflicts,
                    format!(
                        "table {table_ident} is at version {base:?}, not {}",
                        requested - 1
                    ),
                ));
            }
        }

        let current_metadata = current_table.metadata_ref();
        let staged_table = commit.apply(current_table)?;
        if requested_version.is_some()
            && !self.edits_definition(&current_metadata, staged_table.metadata())
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("table {table_ident} definition edit changes nothing"),
            ));
        }
        let staged_metadata_location_str = staged_table.metadata_location_result()?;
        let staged_metadata_location = MetadataLocation::from_str(staged_metadata_location_str)?;

        staged_table
            .metadata()
            .write_to(staged_table.file_io(), &staged_metadata_location)
            .await?;

        let staged_metadata_location_str = staged_metadata_location.to_string();
        let mut tx = self.connection.begin().await.map_err(from_sqlx_error)?;
        let update_result = self
            .execute(
                &format!(
                    "UPDATE {CATALOG_TABLE_NAME}
                     SET {CATALOG_FIELD_METADATA_LOCATION_PROP} = ?, {CATALOG_FIELD_PREVIOUS_METADATA_LOCATION_PROP} = ?
                     WHERE {CATALOG_FIELD_CATALOG_NAME} = ?
                      AND {CATALOG_FIELD_TABLE_NAME} = ?
                      AND {CATALOG_FIELD_TABLE_NAMESPACE} = ?
                      AND (
                        {CATALOG_FIELD_RECORD_TYPE} = '{CATALOG_FIELD_TABLE_RECORD_TYPE}'
                        OR {CATALOG_FIELD_RECORD_TYPE} IS NULL
                      )
                      AND {CATALOG_FIELD_METADATA_LOCATION_PROP} = ?"
                ),
                vec![
                    Some(&staged_metadata_location_str),
                    Some(current_metadata_location.as_str()),
                    Some(&self.name),
                    Some(table_ident.name()),
                    Some(&table_ident.namespace().join(".")),
                    Some(current_metadata_location.as_str()),
                ],
                Some(&mut tx),
            )
            .await?;

        if update_result.rows_affected() == 0 {
            return Err(Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!("Commit conflicted for table: {table_ident}"),
            )
            .with_retryable(true));
        }
        if let Some(requested) = requested_version {
            self.insert_table_version(
                &table_ident,
                staged_table.metadata(),
                requested,
                &staged_metadata_location_str,
                &mut tx,
            )
            .await?;
        }
        self.commit_publication(tx, &table_ident, &staged_metadata_location_str)
            .await?;

        Ok(staged_table)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::hash::Hash;
    use std::sync::Arc;

    use iceberg::io::LocalFsStorageFactory;
    use iceberg::spec::{NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder, Type};
    use iceberg::table::Table;
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    use iceberg::{
        Catalog, CatalogBuilder, ErrorKind, Namespace, NamespaceIdent, TableCreation, TableIdent,
    };
    use itertools::Itertools;
    use regex::Regex;
    use sqlx::Row;
    use sqlx::migrate::MigrateDatabase;
    use tempfile::TempDir;

    use crate::catalog::{
        NAMESPACE_LOCATION_PROPERTY_KEY, SQL_CATALOG_PROP_BIND_STYLE,
        SQL_CATALOG_PROP_TABLE_VERSION, SQL_CATALOG_PROP_URI, SQL_CATALOG_PROP_WAREHOUSE,
    };
    use crate::{SqlBindStyle, SqlCatalog, SqlCatalogBuilder};

    const UUID_REGEX_STR: &str = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";

    fn temp_path() -> String {
        let temp_dir = TempDir::new().unwrap();
        temp_dir.path().to_str().unwrap().to_string()
    }

    fn to_set<T: Eq + Hash>(vec: Vec<T>) -> HashSet<T> {
        HashSet::from_iter(vec)
    }

    fn default_properties() -> HashMap<String, String> {
        HashMap::from([("exists".to_string(), "true".to_string())])
    }

    /// Create a new SQLite catalog for testing. If name is not specified it defaults to "iceberg".
    async fn new_sql_catalog(
        warehouse_location: String,
        name: Option<impl ToString>,
    ) -> SqlCatalog {
        let name = if let Some(name) = name {
            name.to_string()
        } else {
            "iceberg".to_string()
        };
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();

        let props = HashMap::from_iter([
            (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri.to_string()),
            (SQL_CATALOG_PROP_WAREHOUSE.to_string(), warehouse_location),
            (
                SQL_CATALOG_PROP_BIND_STYLE.to_string(),
                SqlBindStyle::DollarNumeric.to_string(),
            ),
        ]);
        SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(&name, props)
            .await
            .unwrap()
    }

    async fn create_namespace<C: Catalog>(catalog: &C, namespace_ident: &NamespaceIdent) {
        let _ = catalog
            .create_namespace(namespace_ident, HashMap::new())
            .await
            .unwrap();
    }

    async fn create_namespaces<C: Catalog>(catalog: &C, namespace_idents: &Vec<&NamespaceIdent>) {
        for namespace_ident in namespace_idents {
            let _ = create_namespace(catalog, namespace_ident).await;
        }
    }

    fn simple_table_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap()
    }

    async fn create_table<C: Catalog>(catalog: &C, table_ident: &TableIdent) {
        let _ = catalog
            .create_table(
                &table_ident.namespace,
                TableCreation::builder()
                    .name(table_ident.name().into())
                    .schema(simple_table_schema())
                    .location(temp_path())
                    .build(),
            )
            .await
            .unwrap();
    }

    async fn create_tables<C: Catalog>(catalog: &C, table_idents: Vec<&TableIdent>) {
        for table_ident in table_idents {
            create_table(catalog, table_ident).await;
        }
    }

    fn assert_table_eq(table: &Table, expected_table_ident: &TableIdent, expected_schema: &Schema) {
        assert_eq!(table.identifier(), expected_table_ident);

        let metadata = table.metadata();

        assert_eq!(metadata.current_schema().as_ref(), expected_schema);

        let expected_partition_spec = PartitionSpec::builder(expected_schema.clone())
            .with_spec_id(0)
            .build()
            .unwrap();

        assert_eq!(
            metadata
                .partition_specs_iter()
                .map(|p| p.as_ref())
                .collect_vec(),
            vec![&expected_partition_spec]
        );

        let expected_sorted_order = SortOrder::builder()
            .with_order_id(0)
            .with_fields(vec![])
            .build(expected_schema)
            .unwrap();

        assert_eq!(
            metadata
                .sort_orders_iter()
                .map(|s| s.as_ref())
                .collect_vec(),
            vec![&expected_sorted_order]
        );

        assert_eq!(metadata.properties(), &HashMap::new());

        assert!(!table.readonly());
    }

    fn assert_table_metadata_location_matches(table: &Table, regex_str: &str) {
        let actual = table.metadata_location().unwrap().to_string();
        let regex = Regex::new(regex_str).unwrap();
        assert!(regex.is_match(&actual))
    }

    #[tokio::test]
    async fn test_initialized() {
        let warehouse_loc = temp_path();
        new_sql_catalog(warehouse_loc.clone(), Some("iceberg")).await;
        // catalog instantiation should not fail even if tables exist
        new_sql_catalog(warehouse_loc.clone(), Some("iceberg")).await;
        new_sql_catalog(warehouse_loc.clone(), Some("iceberg")).await;
    }

    // Regression test: storage-backend props set on the catalog must reach
    // the FileIO; otherwise authenticated backends fail with 401s on writes.
    #[tokio::test]
    async fn test_storage_props_propagate_to_file_io() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        let warehouse_location = temp_path();

        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "iceberg",
                HashMap::from_iter([
                    (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri),
                    (SQL_CATALOG_PROP_WAREHOUSE.to_string(), warehouse_location),
                    ("s3.region".to_string(), "us-east-1".to_string()),
                    ("hf.token".to_string(), "hf_test_token".to_string()),
                ]),
            )
            .await
            .unwrap();

        let props = catalog.fileio.config().props();
        assert_eq!(props.get("s3.region"), Some(&"us-east-1".to_string()));
        assert_eq!(props.get("hf.token"), Some(&"hf_test_token".to_string()));
    }

    #[tokio::test]
    async fn test_builder_method() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        let warehouse_location = temp_path();

        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .uri(sql_lite_uri.to_string())
            .warehouse_location(warehouse_location.clone())
            .sql_bind_style(SqlBindStyle::QMark)
            .load("iceberg", HashMap::default())
            .await;
        assert!(catalog.is_ok());

        let catalog = catalog.unwrap();
        assert!(catalog.warehouse_location == warehouse_location);
        assert!(catalog.sql_bind_style == SqlBindStyle::QMark);
    }

    /// Overwriting an sqlite database with a non-existent path causes
    /// catalog generation to fail
    #[tokio::test]
    async fn test_builder_props_non_existent_path_fails() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        let sql_lite_uri2 = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        let warehouse_location = temp_path();

        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .uri(sql_lite_uri)
            .warehouse_location(warehouse_location)
            .load(
                "iceberg",
                HashMap::from_iter([(SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri2)]),
            )
            .await;
        assert!(catalog.is_err());
    }

    /// Even when an invalid URI is specified in a builder method,
    /// it can be successfully overridden with a valid URI in props
    /// for catalog generation to succeed.
    #[tokio::test]
    async fn test_builder_props_set_valid_uri() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        let sql_lite_uri2 = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        let warehouse_location = temp_path();

        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .uri(sql_lite_uri2)
            .warehouse_location(warehouse_location)
            .load(
                "iceberg",
                HashMap::from_iter([(SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri.clone())]),
            )
            .await;
        assert!(catalog.is_ok());
    }

    /// values assigned via props take precedence
    #[tokio::test]
    async fn test_builder_props_take_precedence() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        let warehouse_location = temp_path();
        let warehouse_location2 = temp_path();

        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .warehouse_location(warehouse_location2)
            .sql_bind_style(SqlBindStyle::DollarNumeric)
            .load(
                "iceberg",
                HashMap::from_iter([
                    (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri),
                    (
                        SQL_CATALOG_PROP_WAREHOUSE.to_string(),
                        warehouse_location.clone(),
                    ),
                    (
                        SQL_CATALOG_PROP_BIND_STYLE.to_string(),
                        SqlBindStyle::QMark.to_string(),
                    ),
                ]),
            )
            .await;

        assert!(catalog.is_ok());

        let catalog = catalog.unwrap();
        assert!(catalog.warehouse_location == warehouse_location);
        assert!(catalog.sql_bind_style == SqlBindStyle::QMark);
    }

    /// values assigned via props take precedence
    #[tokio::test]
    async fn test_builder_props_take_precedence_props() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        let sql_lite_uri2 = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        let warehouse_location = temp_path();
        let warehouse_location2 = temp_path();

        let props = HashMap::from_iter([
            (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri.clone()),
            (
                SQL_CATALOG_PROP_WAREHOUSE.to_string(),
                warehouse_location.clone(),
            ),
            (
                SQL_CATALOG_PROP_BIND_STYLE.to_string(),
                SqlBindStyle::QMark.to_string(),
            ),
        ]);
        let props2 = HashMap::from_iter([
            (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri2.clone()),
            (
                SQL_CATALOG_PROP_WAREHOUSE.to_string(),
                warehouse_location2.clone(),
            ),
            (
                SQL_CATALOG_PROP_BIND_STYLE.to_string(),
                SqlBindStyle::DollarNumeric.to_string(),
            ),
        ]);

        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .props(props2)
            .load("iceberg", props)
            .await;

        assert!(catalog.is_ok());

        let catalog = catalog.unwrap();
        assert!(catalog.warehouse_location == warehouse_location);
        assert!(catalog.sql_bind_style == SqlBindStyle::QMark);
    }

    /// values assigned via props take precedence
    #[tokio::test]
    async fn test_builder_props_take_precedence_prop() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        let sql_lite_uri2 = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        let warehouse_location = temp_path();
        let warehouse_location2 = temp_path();

        let props = HashMap::from_iter([
            (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri.clone()),
            (
                SQL_CATALOG_PROP_WAREHOUSE.to_string(),
                warehouse_location.clone(),
            ),
            (
                SQL_CATALOG_PROP_BIND_STYLE.to_string(),
                SqlBindStyle::QMark.to_string(),
            ),
        ]);

        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .prop(SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri2)
            .prop(SQL_CATALOG_PROP_WAREHOUSE.to_string(), warehouse_location2)
            .prop(
                SQL_CATALOG_PROP_BIND_STYLE.to_string(),
                SqlBindStyle::DollarNumeric.to_string(),
            )
            .load("iceberg", props)
            .await;

        assert!(catalog.is_ok());

        let catalog = catalog.unwrap();
        assert!(catalog.warehouse_location == warehouse_location);
        assert!(catalog.sql_bind_style == SqlBindStyle::QMark);
    }

    /// invalid value for `SqlBindStyle` causes catalog creation to fail
    #[tokio::test]
    async fn test_builder_props_invalid_bind_style_fails() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        let warehouse_location = temp_path();

        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "iceberg",
                HashMap::from_iter([
                    (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri),
                    (SQL_CATALOG_PROP_WAREHOUSE.to_string(), warehouse_location),
                    (SQL_CATALOG_PROP_BIND_STYLE.to_string(), "AAA".to_string()),
                ]),
            )
            .await;

        assert!(catalog.is_err());
    }

    #[tokio::test]
    async fn test_idle_gated_pool_serves_reused_connections() {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        // One connection and a zero threshold force every reuse through the
        // idle-gated ping.
        let catalog = SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "iceberg",
                HashMap::from_iter([
                    (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri),
                    (SQL_CATALOG_PROP_WAREHOUSE.to_string(), temp_path()),
                    ("pool.max-connections".to_string(), "1".to_string()),
                    ("pool.test-idle-after-ms".to_string(), "0".to_string()),
                ]),
            )
            .await
            .unwrap();

        for _ in 0..3 {
            assert_eq!(catalog.list_namespaces(None).await.unwrap(), vec![]);
        }
    }

    #[tokio::test]
    async fn test_builder_props_invalid_pool_property_fails() {
        for property in [
            "pool.max-connections",
            "pool.idle-timeout",
            "pool.test-before-acquire",
            "pool.test-idle-after-ms",
        ] {
            let error = SqlCatalogBuilder::default()
                .with_storage_factory(Arc::new(LocalFsStorageFactory))
                .prop(property, "invalid")
                .load("iceberg", HashMap::new())
                .await
                .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::DataInvalid);
            assert!(error.to_string().contains(property));
            assert!(error.to_string().contains("invalid"));
        }
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_empty_vector() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;

        assert_eq!(catalog.list_namespaces(None).await.unwrap(), vec![]);
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_empty_different_name() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc.clone(), Some("iceberg")).await;
        let namespace_ident_1 = NamespaceIdent::new("a".into());
        let namespace_ident_2 = NamespaceIdent::new("b".into());
        create_namespaces(&catalog, &vec![&namespace_ident_1, &namespace_ident_2]).await;
        assert_eq!(
            to_set(catalog.list_namespaces(None).await.unwrap()),
            to_set(vec![namespace_ident_1, namespace_ident_2])
        );

        let catalog2 = new_sql_catalog(warehouse_loc, Some("test")).await;
        assert_eq!(catalog2.list_namespaces(None).await.unwrap(), vec![]);
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_only_top_level_namespaces() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_1 = NamespaceIdent::new("a".into());
        let namespace_ident_2 = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_3 = NamespaceIdent::new("b".into());
        create_namespaces(
            &catalog,
            &vec![&namespace_ident_1, &namespace_ident_2, &namespace_ident_3],
        )
        .await;

        assert_eq!(
            to_set(catalog.list_namespaces(None).await.unwrap()),
            to_set(vec![namespace_ident_1, namespace_ident_3])
        );
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_no_namespaces_under_parent() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_1 = NamespaceIdent::new("a".into());
        let namespace_ident_2 = NamespaceIdent::new("b".into());
        create_namespaces(&catalog, &vec![&namespace_ident_1, &namespace_ident_2]).await;

        assert_eq!(
            catalog
                .list_namespaces(Some(&namespace_ident_1))
                .await
                .unwrap(),
            vec![]
        );
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_namespace_under_parent() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_1 = NamespaceIdent::new("a".into());
        let namespace_ident_2 = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_3 = NamespaceIdent::new("c".into());
        create_namespaces(
            &catalog,
            &vec![&namespace_ident_1, &namespace_ident_2, &namespace_ident_3],
        )
        .await;

        assert_eq!(
            to_set(catalog.list_namespaces(None).await.unwrap()),
            to_set(vec![namespace_ident_1.clone(), namespace_ident_3])
        );

        assert_eq!(
            catalog
                .list_namespaces(Some(&namespace_ident_1))
                .await
                .unwrap(),
            vec![NamespaceIdent::from_strs(vec!["a", "b"]).unwrap()]
        );
    }

    #[tokio::test]
    async fn test_list_namespaces_returns_multiple_namespaces_under_parent() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_1 = NamespaceIdent::new("a".to_string());
        let namespace_ident_2 = NamespaceIdent::from_strs(vec!["a", "a"]).unwrap();
        let namespace_ident_3 = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_4 = NamespaceIdent::from_strs(vec!["a", "c"]).unwrap();
        let namespace_ident_5 = NamespaceIdent::new("b".into());
        create_namespaces(
            &catalog,
            &vec![
                &namespace_ident_1,
                &namespace_ident_2,
                &namespace_ident_3,
                &namespace_ident_4,
                &namespace_ident_5,
            ],
        )
        .await;

        assert_eq!(
            to_set(
                catalog
                    .list_namespaces(Some(&namespace_ident_1))
                    .await
                    .unwrap()
            ),
            to_set(vec![
                NamespaceIdent::from_strs(vec!["a", "a"]).unwrap(),
                NamespaceIdent::from_strs(vec!["a", "b"]).unwrap(),
                NamespaceIdent::from_strs(vec!["a", "c"]).unwrap(),
            ])
        );
    }

    #[tokio::test]
    async fn test_namespace_exists_returns_false() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        assert!(
            !catalog
                .namespace_exists(&NamespaceIdent::new("b".into()))
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_namespace_exists_returns_true() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        assert!(catalog.namespace_exists(&namespace_ident).await.unwrap());
    }

    #[tokio::test]
    async fn test_create_namespace_with_properties() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::new("abc".into());

        let mut properties = default_properties();
        properties.insert("k".into(), "v".into());

        assert_eq!(
            catalog
                .create_namespace(&namespace_ident, properties.clone())
                .await
                .unwrap(),
            Namespace::with_properties(namespace_ident.clone(), properties.clone())
        );

        assert_eq!(
            catalog.get_namespace(&namespace_ident).await.unwrap(),
            Namespace::with_properties(namespace_ident, properties)
        );
    }

    #[tokio::test]
    async fn test_create_nested_namespace() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let parent_namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &parent_namespace_ident).await;

        let child_namespace_ident = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();

        assert_eq!(
            catalog
                .create_namespace(&child_namespace_ident, HashMap::new())
                .await
                .unwrap(),
            Namespace::new(child_namespace_ident.clone())
        );

        assert_eq!(
            catalog.get_namespace(&child_namespace_ident).await.unwrap(),
            Namespace::with_properties(child_namespace_ident, default_properties())
        );
    }

    #[tokio::test]
    async fn test_create_deeply_nested_namespace() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();

        assert_eq!(
            catalog
                .create_namespace(&namespace_ident_a_b_c, HashMap::new())
                .await
                .unwrap(),
            Namespace::new(namespace_ident_a_b_c.clone())
        );

        assert_eq!(
            catalog.get_namespace(&namespace_ident_a_b_c).await.unwrap(),
            Namespace::with_properties(namespace_ident_a_b_c, default_properties())
        );
    }

    #[tokio::test]
    async fn test_update_namespace_noop() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        catalog
            .update_namespace(&namespace_ident, HashMap::new())
            .await
            .unwrap();

        assert_eq!(
            *catalog
                .get_namespace(&namespace_ident)
                .await
                .unwrap()
                .properties(),
            HashMap::from_iter([("exists".to_string(), "true".to_string())])
        )
    }

    #[tokio::test]
    async fn test_update_nested_namespace() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::from_strs(["a", "b"]).unwrap();
        create_namespace(&catalog, &namespace_ident).await;

        let mut props = HashMap::from_iter([
            ("prop1".to_string(), "val1".to_string()),
            ("prop2".into(), "val2".into()),
        ]);

        catalog
            .update_namespace(&namespace_ident, props.clone())
            .await
            .unwrap();

        props.insert("exists".into(), "true".into());

        assert_eq!(
            *catalog
                .get_namespace(&namespace_ident)
                .await
                .unwrap()
                .properties(),
            props
        )
    }

    #[tokio::test]
    async fn test_update_namespace_errors_if_nested_namespace_doesnt_exist() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::from_strs(["a", "b"]).unwrap();

        let props = HashMap::from_iter([
            ("prop1".to_string(), "val1".to_string()),
            ("prop2".into(), "val2".into()),
        ]);

        let err = catalog
            .update_namespace(&namespace_ident, props)
            .await
            .unwrap_err();

        assert_eq!(
            err.message(),
            format!("No such namespace: {namespace_ident:?}")
        );
    }

    #[tokio::test]
    async fn test_drop_nested_namespace() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        catalog.drop_namespace(&namespace_ident_a_b).await.unwrap();

        assert!(
            !catalog
                .namespace_exists(&namespace_ident_a_b)
                .await
                .unwrap()
        );

        assert!(catalog.namespace_exists(&namespace_ident_a).await.unwrap());
    }

    #[tokio::test]
    async fn test_drop_deeply_nested_namespace() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();
        create_namespaces(
            &catalog,
            &vec![
                &namespace_ident_a,
                &namespace_ident_a_b,
                &namespace_ident_a_b_c,
            ],
        )
        .await;

        catalog
            .drop_namespace(&namespace_ident_a_b_c)
            .await
            .unwrap();

        assert!(
            !catalog
                .namespace_exists(&namespace_ident_a_b_c)
                .await
                .unwrap()
        );

        assert!(
            catalog
                .namespace_exists(&namespace_ident_a_b)
                .await
                .unwrap()
        );

        assert!(catalog.namespace_exists(&namespace_ident_a).await.unwrap());
    }

    #[tokio::test]
    async fn test_drop_namespace_throws_error_if_nested_namespace_doesnt_exist() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        create_namespace(&catalog, &NamespaceIdent::new("a".into())).await;

        let non_existent_namespace_ident =
            NamespaceIdent::from_vec(vec!["a".into(), "b".into()]).unwrap();
        assert_eq!(
            catalog
                .drop_namespace(&non_existent_namespace_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_namespace_ident:?}")
        )
    }

    #[tokio::test]
    async fn test_dropping_a_namespace_does_not_drop_namespaces_nested_under_that_one() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespaces(&catalog, &vec![&namespace_ident_a, &namespace_ident_a_b]).await;

        catalog.drop_namespace(&namespace_ident_a).await.unwrap();

        assert!(!catalog.namespace_exists(&namespace_ident_a).await.unwrap());

        assert!(
            catalog
                .namespace_exists(&namespace_ident_a_b)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_create_table_falls_back_to_namespace_location_if_table_location_is_missing() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;

        let namespace_ident = NamespaceIdent::new("a".into());
        let mut namespace_properties = HashMap::new();
        let namespace_location = temp_path();
        namespace_properties.insert(
            NAMESPACE_LOCATION_PROPERTY_KEY.to_string(),
            namespace_location.to_string(),
        );
        catalog
            .create_namespace(&namespace_ident, namespace_properties)
            .await
            .unwrap();

        let table_name = "tbl1";
        let expected_table_ident = TableIdent::new(namespace_ident.clone(), table_name.into());
        let expected_table_metadata_location_regex =
            format!("^{namespace_location}/tbl1/metadata/00000-{UUID_REGEX_STR}.metadata.json$",);

        let table = catalog
            .create_table(
                &namespace_ident,
                TableCreation::builder()
                    .name(table_name.into())
                    .schema(simple_table_schema())
                    // no location specified for table
                    .build(),
            )
            .await
            .unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);

        let table = catalog.load_table(&expected_table_ident).await.unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);
    }

    #[tokio::test]
    async fn test_create_table_in_nested_namespace_falls_back_to_nested_namespace_location_if_table_location_is_missing()
     {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;

        let namespace_ident = NamespaceIdent::new("a".into());
        let mut namespace_properties = HashMap::new();
        let namespace_location = temp_path();
        namespace_properties.insert(
            NAMESPACE_LOCATION_PROPERTY_KEY.to_string(),
            namespace_location.to_string(),
        );
        catalog
            .create_namespace(&namespace_ident, namespace_properties)
            .await
            .unwrap();

        let nested_namespace_ident = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let mut nested_namespace_properties = HashMap::new();
        let nested_namespace_location = temp_path();
        nested_namespace_properties.insert(
            NAMESPACE_LOCATION_PROPERTY_KEY.to_string(),
            nested_namespace_location.to_string(),
        );
        catalog
            .create_namespace(&nested_namespace_ident, nested_namespace_properties)
            .await
            .unwrap();

        let table_name = "tbl1";
        let expected_table_ident =
            TableIdent::new(nested_namespace_ident.clone(), table_name.into());
        let expected_table_metadata_location_regex = format!(
            "^{nested_namespace_location}/tbl1/metadata/00000-{UUID_REGEX_STR}.metadata.json$",
        );

        let table = catalog
            .create_table(
                &nested_namespace_ident,
                TableCreation::builder()
                    .name(table_name.into())
                    .schema(simple_table_schema())
                    // no location specified for table
                    .build(),
            )
            .await
            .unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);

        let table = catalog.load_table(&expected_table_ident).await.unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);
    }

    #[tokio::test]
    async fn test_create_table_falls_back_to_warehouse_location_if_both_table_location_and_namespace_location_are_missing()
     {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc.clone(), Some("iceberg")).await;

        let namespace_ident = NamespaceIdent::new("a".into());
        // note: no location specified in namespace_properties
        let namespace_properties = HashMap::new();
        catalog
            .create_namespace(&namespace_ident, namespace_properties)
            .await
            .unwrap();

        let table_name = "tbl1";
        let expected_table_ident = TableIdent::new(namespace_ident.clone(), table_name.into());
        let expected_table_metadata_location_regex =
            format!("^{warehouse_loc}/a/tbl1/metadata/00000-{UUID_REGEX_STR}.metadata.json$");

        let table = catalog
            .create_table(
                &namespace_ident,
                TableCreation::builder()
                    .name(table_name.into())
                    .schema(simple_table_schema())
                    // no location specified for table
                    .build(),
            )
            .await
            .unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);

        let table = catalog.load_table(&expected_table_ident).await.unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);
    }

    #[tokio::test]
    async fn test_create_table_in_nested_namespace_falls_back_to_warehouse_location_if_both_table_location_and_namespace_location_are_missing()
     {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc.clone(), Some("iceberg")).await;

        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;

        let nested_namespace_ident = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        create_namespace(&catalog, &nested_namespace_ident).await;

        let table_name = "tbl1";
        let expected_table_ident =
            TableIdent::new(nested_namespace_ident.clone(), table_name.into());
        let expected_table_metadata_location_regex =
            format!("^{warehouse_loc}/a/b/tbl1/metadata/00000-{UUID_REGEX_STR}.metadata.json$");

        let table = catalog
            .create_table(
                &nested_namespace_ident,
                TableCreation::builder()
                    .name(table_name.into())
                    .schema(simple_table_schema())
                    // no location specified for table
                    .build(),
            )
            .await
            .unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);

        let table = catalog.load_table(&expected_table_ident).await.unwrap();
        assert_table_eq(&table, &expected_table_ident, &simple_table_schema());
        assert_table_metadata_location_matches(&table, &expected_table_metadata_location_regex);
    }

    #[tokio::test]
    async fn test_create_table_throws_error_if_table_with_same_name_already_exists() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc.clone(), Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace_ident).await;
        let table_name = "tbl1";
        let table_ident = TableIdent::new(namespace_ident.clone(), table_name.into());
        create_table(&catalog, &table_ident).await;

        let tmp_dir = TempDir::new().unwrap();
        let location = tmp_dir.path().to_str().unwrap().to_string();

        assert_eq!(
            catalog
                .create_table(
                    &namespace_ident,
                    TableCreation::builder()
                        .name(table_name.into())
                        .schema(simple_table_schema())
                        .location(location)
                        .build()
                )
                .await
                .unwrap_err()
                .to_string(),
            format!(
                "TableAlreadyExists => Table {:?} already exists.",
                &table_ident
            )
        );
    }

    #[tokio::test]
    async fn test_rename_table_src_table_is_same_as_dst_table() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        create_namespace(&catalog, &namespace_ident).await;
        let table_ident = TableIdent::new(namespace_ident.clone(), "tbl".into());
        create_table(&catalog, &table_ident).await;

        catalog
            .rename_table(&table_ident, &table_ident)
            .await
            .unwrap();

        assert_eq!(
            catalog.list_tables(&namespace_ident).await.unwrap(),
            vec![table_ident],
        );
    }

    #[tokio::test]
    async fn test_rename_table_across_nested_namespaces() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident_a = NamespaceIdent::new("a".into());
        let namespace_ident_a_b = NamespaceIdent::from_strs(vec!["a", "b"]).unwrap();
        let namespace_ident_a_b_c = NamespaceIdent::from_strs(vec!["a", "b", "c"]).unwrap();
        create_namespaces(
            &catalog,
            &vec![
                &namespace_ident_a,
                &namespace_ident_a_b,
                &namespace_ident_a_b_c,
            ],
        )
        .await;

        let src_table_ident = TableIdent::new(namespace_ident_a_b_c.clone(), "tbl1".into());
        create_tables(&catalog, vec![&src_table_ident]).await;

        let dst_table_ident = TableIdent::new(namespace_ident_a_b.clone(), "tbl1".into());
        catalog
            .rename_table(&src_table_ident, &dst_table_ident)
            .await
            .unwrap();

        assert!(!catalog.table_exists(&src_table_ident).await.unwrap());

        assert!(catalog.table_exists(&dst_table_ident).await.unwrap());
    }

    #[tokio::test]
    async fn test_rename_table_throws_error_if_dst_namespace_doesnt_exist() {
        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let src_namespace_ident = NamespaceIdent::new("n1".into());
        let src_table_ident = TableIdent::new(src_namespace_ident.clone(), "tbl1".into());
        create_namespace(&catalog, &src_namespace_ident).await;
        create_table(&catalog, &src_table_ident).await;

        let non_existent_dst_namespace_ident = NamespaceIdent::new("n2".into());
        let dst_table_ident =
            TableIdent::new(non_existent_dst_namespace_ident.clone(), "tbl1".into());
        assert_eq!(
            catalog
                .rename_table(&src_table_ident, &dst_table_ident)
                .await
                .unwrap_err()
                .to_string(),
            format!("NamespaceNotFound => No such namespace: {non_existent_dst_namespace_ident:?}"),
        );
    }

    #[tokio::test]
    async fn test_load_metadata_location_tracks_each_commit() {
        use iceberg::transaction::{ApplyTransactionAction, Transaction};

        let warehouse_loc = temp_path();
        let catalog = new_sql_catalog(warehouse_loc, Some("iceberg")).await;
        let namespace_ident = NamespaceIdent::new("n1".into());
        let table_ident = TableIdent::new(namespace_ident.clone(), "tbl1".into());
        let missing_ident = TableIdent::new(namespace_ident.clone(), "missing".into());
        create_namespace(&catalog, &namespace_ident).await;
        create_table(&catalog, &table_ident).await;

        let table = catalog.load_table(&table_ident).await.unwrap();
        let created = catalog
            .load_metadata_location(&table_ident)
            .await
            .unwrap()
            .expect("created table has a pointer");
        assert_eq!(table.metadata_location(), Some(created.as_str()));

        let tx = Transaction::new(&table);
        let tx = tx
            .update_table_properties()
            .set("k".to_string(), "v".to_string())
            .apply(tx)
            .unwrap();
        let committed = tx.commit(&catalog).await.unwrap();
        let updated = catalog
            .load_metadata_location(&table_ident)
            .await
            .unwrap()
            .expect("committed table has a pointer");
        assert_ne!(updated, created, "a commit writes a new metadata location");
        assert_eq!(committed.metadata_location(), Some(updated.as_str()));

        assert_eq!(
            catalog
                .load_metadata_location(&missing_ident)
                .await
                .unwrap(),
            None
        );
    }

    const TABLE_VERSION: &str = "wyrd.bifrost.table-version";

    /// Creates a SQLite catalog that versions table definitions under
    /// [`TABLE_VERSION`].
    async fn new_versioned_catalog() -> SqlCatalog {
        let sql_lite_uri = format!("sqlite:{}", temp_path());
        sqlx::Sqlite::create_database(&sql_lite_uri).await.unwrap();
        SqlCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "iceberg",
                HashMap::from_iter([
                    (SQL_CATALOG_PROP_URI.to_string(), sql_lite_uri),
                    (SQL_CATALOG_PROP_WAREHOUSE.to_string(), temp_path()),
                    (
                        SQL_CATALOG_PROP_TABLE_VERSION.to_string(),
                        TABLE_VERSION.to_string(),
                    ),
                ]),
            )
            .await
            .unwrap()
    }

    /// Creates one versioned table and returns it.
    async fn create_versioned_table(catalog: &SqlCatalog) -> Table {
        let namespace = NamespaceIdent::new("a".into());
        create_namespace(catalog, &namespace).await;
        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("tbl".into())
                    .schema(simple_table_schema())
                    .location(temp_path())
                    .build(),
            )
            .await
            .unwrap()
    }

    /// Commits `properties` on `table` with one catalog update attempt.
    async fn set_properties(
        catalog: &SqlCatalog,
        table: &Table,
        properties: &[(&str, &str)],
    ) -> iceberg::Result<Table> {
        let tx = Transaction::new(table);
        let action = properties
            .iter()
            .fold(tx.update_table_properties(), |action, (key, value)| {
                action.set((*key).to_string(), (*value).to_string())
            });
        action.apply(tx)?.commit_once(catalog).await
    }

    /// Lists every retained `(table_version, metadata_location)` reference.
    async fn retained_versions(catalog: &SqlCatalog) -> Vec<(i64, String)> {
        sqlx::query("SELECT table_version, metadata_location FROM iceberg_table_versions ORDER BY table_version")
            .fetch_all(&catalog.connection)
            .await
            .unwrap()
            .iter()
            .map(|row| (row.get(0), row.get(1)))
            .collect()
    }

    #[tokio::test]
    async fn versioned_creation_publishes_version_one() {
        let catalog = new_versioned_catalog().await;
        let table = create_versioned_table(&catalog).await;
        let location = table.metadata_location().unwrap().to_string();

        assert_eq!(
            table.metadata().properties().get(TABLE_VERSION),
            Some(&"1".to_string())
        );
        assert_eq!(retained_versions(&catalog).await, vec![(
            1,
            location.clone()
        )]);
        assert_eq!(
            catalog
                .load_table_version_location(
                    table.identifier(),
                    &table.metadata().uuid().to_string(),
                    1
                )
                .await
                .unwrap(),
            Some(location)
        );
        assert_eq!(
            catalog
                .load_table_version_location(
                    table.identifier(),
                    &table.metadata().uuid().to_string(),
                    2
                )
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn versioned_creation_refuses_any_other_initial_version() {
        let catalog = new_versioned_catalog().await;
        let namespace = NamespaceIdent::new("a".into());
        create_namespace(&catalog, &namespace).await;

        let refused = catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("tbl".into())
                    .schema(simple_table_schema())
                    .location(temp_path())
                    .properties(HashMap::from([(TABLE_VERSION.into(), "2".into())]))
                    .build(),
            )
            .await;

        assert_eq!(refused.unwrap_err().kind(), ErrorKind::DataInvalid);
        assert!(retained_versions(&catalog).await.is_empty());
    }

    #[tokio::test]
    async fn a_definition_commit_publishes_the_next_version_with_its_pointer() {
        let catalog = new_versioned_catalog().await;
        let table = create_versioned_table(&catalog).await;
        let first = table.metadata_location().unwrap().to_string();

        let data_only = set_properties(&catalog, &table, &[("other", "x")])
            .await
            .unwrap();
        assert_eq!(retained_versions(&catalog).await.len(), 1);

        let edited = set_properties(&catalog, &data_only, &[
            ("bloom", "foo"),
            (TABLE_VERSION, "2"),
        ])
        .await
        .unwrap();
        let second = edited.metadata_location().unwrap().to_string();

        assert_eq!(retained_versions(&catalog).await, vec![
            (1, first.clone()),
            (2, second.clone())
        ]);
        let uuid = table.metadata().uuid().to_string();
        assert_eq!(
            catalog
                .list_table_version_locations(table.identifier(), &uuid)
                .await
                .unwrap(),
            vec![first, second.clone()]
        );
        assert!(
            catalog
                .list_table_version_locations(
                    table.identifier(),
                    "00000000-0000-0000-0000-000000000000"
                )
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            catalog
                .load_metadata_location(table.identifier())
                .await
                .unwrap(),
            Some(second)
        );
    }

    #[tokio::test]
    async fn competing_definition_commits_from_one_version_yield_one_winner() {
        let catalog = new_versioned_catalog().await;
        let base = create_versioned_table(&catalog).await;

        set_properties(&catalog, &base, &[
            ("bloom", "winner"),
            (TABLE_VERSION, "2"),
        ])
        .await
        .unwrap();
        let stale = set_properties(&catalog, &base, &[("bloom", "loser"), (TABLE_VERSION, "2")])
            .await
            .unwrap_err();

        assert_eq!(stale.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(!stale.retryable(), "a stale definition edit never rebases");
        assert_eq!(retained_versions(&catalog).await.len(), 2);
        let current = catalog.load_table(base.identifier()).await.unwrap();
        assert_eq!(
            current.metadata().properties().get("bloom"),
            Some(&"winner".to_string())
        );
    }

    #[tokio::test]
    async fn data_only_head_movement_keeps_a_pending_definition_edit() {
        let catalog = new_versioned_catalog().await;
        let base = create_versioned_table(&catalog).await;
        set_properties(&catalog, &base, &[("other", "x")])
            .await
            .unwrap();

        let edited = set_properties(&catalog, &base, &[("bloom", "foo"), (TABLE_VERSION, "2")])
            .await
            .unwrap();

        let properties = edited.metadata().properties();
        assert_eq!(properties.get("other"), Some(&"x".to_string()));
        assert_eq!(properties.get("bloom"), Some(&"foo".to_string()));
        assert_eq!(properties.get(TABLE_VERSION), Some(&"2".to_string()));
        assert_eq!(retained_versions(&catalog).await.len(), 2);
    }

    #[tokio::test]
    async fn invalid_version_changes_are_refused_without_a_reference() {
        let catalog = new_versioned_catalog().await;
        let table = create_versioned_table(&catalog).await;

        for properties in [
            vec![(TABLE_VERSION, "2")],
            vec![("bloom", "foo"), (TABLE_VERSION, "3")],
            vec![("bloom", "foo"), (TABLE_VERSION, "02")],
            vec![("bloom", "foo"), (TABLE_VERSION, "+2")],
            vec![("bloom", "foo"), (TABLE_VERSION, "9223372036854775808")],
        ] {
            assert!(
                set_properties(&catalog, &table, &properties).await.is_err(),
                "{properties:?} must be refused"
            );
        }
        let removal = Transaction::new(&table);
        let removed = removal
            .update_table_properties()
            .set("bloom".into(), "foo".into())
            .remove(TABLE_VERSION.into())
            .apply(removal)
            .unwrap()
            .commit_once(&catalog)
            .await;

        assert!(removed.is_err(), "the table version cannot be removed");
        let unversioned = Transaction::new(&table);
        let unversioned = unversioned
            .replace_sort_order()
            .asc("foo", iceberg::spec::NullOrder::First)
            .apply(unversioned)
            .unwrap()
            .commit_once(&catalog)
            .await;
        assert!(
            unversioned.is_err(),
            "a native definition edit must advance the table version"
        );
        assert_eq!(retained_versions(&catalog).await.len(), 1);

        let bloom = r#"["a","b"]"#;
        let edited = set_properties(&catalog, &table, &[("bloom", bloom), (TABLE_VERSION, "2")])
            .await
            .unwrap();
        let pointer = edited.metadata_location().map(str::to_owned);
        assert!(
            set_properties(&catalog, &edited, &[("bloom", bloom), (TABLE_VERSION, "3")])
                .await
                .is_err(),
            "a version-intended edit that changes nothing is refused"
        );
        let current = catalog.load_table(edited.identifier()).await.unwrap();
        assert_eq!(current.metadata_location().map(str::to_owned), pointer);
        assert_eq!(
            current.metadata().properties().get(TABLE_VERSION),
            Some(&"2".to_string())
        );
        assert_eq!(retained_versions(&catalog).await.len(), 2);
        set_properties(&catalog, &edited, &[
            ("bloom", r#"["b","a"]"#),
            (TABLE_VERSION, "3"),
        ])
        .await
        .expect("a reordered Bloom list is an effective edit");
        assert_eq!(retained_versions(&catalog).await.len(), 3);
    }
}
