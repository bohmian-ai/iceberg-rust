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

//! Variant fields at any Struct or List depth, and arrays rebuilt by name.
//!
//! A Variant field is addressed by its name path from a top-level column:
//! the names of the Struct children leading to it, with one step for each
//! List element it sits in (the element field's name, which is never
//! checked). Writers shred, and readers unshred, through
//! [`map_variant_field`]; [`rebuild_by_name`] restores a column read with
//! fewer children to its full type.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, ListArray, StructArray, make_array, new_null_array};
use arrow_buffer::NullBuffer;
use arrow_cast::cast;
use arrow_schema::extension::ExtensionType;
use arrow_schema::{DataType, Field, FieldRef, Fields};
use parquet::variant::{VariantArray, VariantType, unshred_variant};

use crate::{Error, ErrorKind, Result};

/// Returns whether `field` is a Variant (by its extension name).
pub fn is_variant_field(field: &Field) -> bool {
    field.extension_type_name() == Some(VariantType::NAME)
}

/// Returns the name path of every Variant field in `fields`, at any Struct
/// or List depth, in schema order.
///
/// A Variant is a leaf of the walk: Variants are never nested in a
/// Variant's storage. Maps are not entered.
pub fn variant_field_paths(fields: &Fields) -> Vec<Vec<String>> {
    let mut paths = Vec::new();
    let mut path = Vec::new();
    for field in fields {
        collect_variant_paths(field, &mut path, &mut paths);
    }
    paths
}

/// Adds the paths of the Variant fields at or below `field` to `paths`.
fn collect_variant_paths(field: &Field, path: &mut Vec<String>, paths: &mut Vec<Vec<String>>) {
    path.push(field.name().clone());
    if is_variant_field(field) {
        paths.push(path.clone());
    } else {
        match field.data_type() {
            DataType::Struct(children) => {
                for child in children {
                    collect_variant_paths(child, path, paths);
                }
            }
            DataType::List(element) => collect_variant_paths(element, path, paths),
            _ => {}
        }
    }
    path.pop();
}

/// Replaces the Variant array at `path` inside `array` with `f` of it, and
/// returns `field` retyped to the result and the array rebuilt around it.
///
/// `path` starts with the column's own name. The walk follows `array`'s own
/// type: a Struct step finds the child by name and a List step enters the
/// element, so `field` may be the logical field of a narrower read. A step
/// the array lacks, because the read skipped it, leaves the array unchanged.
/// `f` sees every value at that position, including ones under a null
/// parent, so it must be a per-value transformation; parent validity and
/// List offsets are kept.
///
/// # Errors
///
/// Returns the error `f` returns, or the Arrow error raised when a rebuilt
/// array is invalid.
pub fn map_variant_field(
    field: &FieldRef,
    array: &ArrayRef,
    path: &[String],
    f: &mut dyn FnMut(&ArrayRef) -> Result<ArrayRef>,
) -> Result<(FieldRef, ArrayRef)> {
    let array = map_at(array, path.get(1..).unwrap_or_default(), f)?;
    Ok((
        Arc::new(
            field
                .as_ref()
                .clone()
                .with_data_type(array.data_type().clone()),
        ),
        array,
    ))
}

/// [`map_variant_field`] below `array`, at the path steps after its own name.
fn map_at(
    array: &ArrayRef,
    rest: &[String],
    f: &mut dyn FnMut(&ArrayRef) -> Result<ArrayRef>,
) -> Result<ArrayRef> {
    let Some((step, tail)) = rest.split_first() else {
        return f(array);
    };
    match array.data_type() {
        DataType::List(element) => {
            let list = array.as_list::<i32>();
            let values = map_at(list.values(), tail, f)?;
            let element = Arc::new(
                element
                    .as_ref()
                    .clone()
                    .with_data_type(values.data_type().clone()),
            );
            Ok(Arc::new(ListArray::try_new(
                element,
                list.offsets().clone(),
                values,
                list.nulls().cloned(),
            )?))
        }
        DataType::Struct(children) => {
            let Some((index, child)) = children.find(step) else {
                return Ok(Arc::clone(array));
            };
            let source = array.as_struct();
            let column = map_at(source.column(index), tail, f)?;
            let mut fields: Vec<FieldRef> = children.iter().cloned().collect();
            fields[index] = Arc::new(
                child
                    .as_ref()
                    .clone()
                    .with_data_type(column.data_type().clone()),
            );
            let mut columns = source.columns().to_vec();
            columns[index] = column;
            Ok(Arc::new(StructArray::try_new(
                Fields::from(fields),
                columns,
                source.nulls().cloned(),
            )?))
        }
        _ => Ok(Arc::clone(array)),
    }
}

/// Applies `f` to every Variant field `field` declares, at any Struct or
/// List depth, inside `array`, through [`map_variant_field`].
///
/// The Variant fields are found from `field`, so a caller passing a
/// column's logical field finds them even when `array`'s own types carry no
/// Variant marker.
///
/// # Errors
///
/// Returns the error `f` returns.
pub fn map_variant_fields(
    field: &FieldRef,
    array: &ArrayRef,
    f: &mut dyn FnMut(&ArrayRef) -> Result<ArrayRef>,
) -> Result<(FieldRef, ArrayRef)> {
    let mut field = Arc::clone(field);
    let mut array = Arc::clone(array);
    for path in variant_field_paths(&Fields::from(vec![Arc::clone(&field)])) {
        (field, array) = map_variant_field(&field, &array, &path, f)?;
    }
    Ok((field, array))
}

/// Returns `field` and `array` with every shredded Variant at any Struct or
/// List depth restored to its logical `metadata`/`value` storage.
///
/// A shredded Variant is one whose storage has a `typed_value`; it may hold
/// only some of its file's shredded children, when a read skipped the rest,
/// and still unshreds to the values of the children it has. Anything else
/// is returned unchanged.
///
/// # Errors
///
/// Returns an error when a shredded Variant is not valid Variant storage or
/// Arrow cannot unshred it.
pub fn unshred_variants(field: &FieldRef, array: &ArrayRef) -> Result<(FieldRef, ArrayRef)> {
    map_variant_fields(field, array, &mut |variant| {
        let DataType::Struct(children) = variant.data_type() else {
            return Ok(Arc::clone(variant));
        };
        if children.find("typed_value").is_none() {
            return Ok(Arc::clone(variant));
        }
        let variant_error = |err| {
            Error::new(ErrorKind::DataInvalid, "Failed to unshred variant column").with_source(err)
        };
        let variant = VariantArray::try_new(variant.as_ref()).map_err(variant_error)?;
        Ok(unshred_variant(&variant).map_err(variant_error)?.into())
    })
}

/// Returns `source` at `target_type`, rebuilding Structs by child name.
///
/// A Struct is rebuilt with the target's children: each child `source` has
/// is rebuilt in turn, and each child it lacks is null when nullable. A required child is only missing when a read skipped
/// it, since a required field cannot be added without a default; that child
/// holds zeroed values the read's caller never uses. A List is rebuilt
/// around its rebuilt element. An equal type is returned as is; everything
/// else is an Arrow cast.
///
/// # Errors
///
/// Returns the Arrow error raised by a cast or an invalid rebuilt array.
pub fn rebuild_by_name(source: &ArrayRef, target_type: &DataType) -> Result<ArrayRef> {
    match (source.data_type(), target_type) {
        (read, target) if read == target => Ok(Arc::clone(source)),
        (DataType::List(_), DataType::List(target)) => {
            let list = source.as_list::<i32>();
            let values = rebuild_by_name(list.values(), target.data_type())?;
            Ok(Arc::new(ListArray::try_new(
                Arc::clone(target),
                list.offsets().clone(),
                values,
                list.nulls().cloned(),
            )?))
        }
        (DataType::Struct(_), DataType::Struct(target)) => {
            let source = source.as_struct();
            let children = target
                .iter()
                .map(|field| match source.column_by_name(field.name()) {
                    Some(child) => rebuild_by_name(child, field.data_type()),
                    None => placeholder(field, source.len()),
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(StructArray::try_new(
                target.clone(),
                children,
                source.nulls().cloned(),
            )?))
        }
        _ => Ok(cast(source, target_type)?),
    }
}

/// A never-read stand-in for `field` of `len` rows: all null, or for a
/// required field zeroed values with no null buffer. A Struct is built
/// child by child, so its required children are zeroed rather than null.
///
/// # Errors
///
/// Returns the Arrow error raised when the stand-in is invalid.
fn placeholder(field: &Field, len: usize) -> Result<ArrayRef> {
    if let DataType::Struct(children) = field.data_type() {
        let columns = children
            .iter()
            .map(|child| placeholder(child, len))
            .collect::<Result<Vec<_>>>()?;
        let nulls = field.is_nullable().then(|| NullBuffer::new_null(len));
        return Ok(Arc::new(StructArray::try_new(
            children.clone(),
            columns,
            nulls,
        )?));
    }
    let nulls = new_null_array(field.data_type(), len);
    if field.is_nullable() {
        return Ok(nulls);
    }
    Ok(make_array(
        nulls.into_data().into_builder().nulls(None).build()?,
    ))
}

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use arrow_buffer::OffsetBuffer;
    use parquet::variant::{json_to_variant, shred_variant, variant_to_json};

    use super::*;

    /// A `List<Struct<name: Utf8, attributes: Variant>>` field and array of
    /// two rows, `[{a,1},{b,2}]` and `[{c,3}]`, attributes `{"k": n}`.
    fn events() -> (FieldRef, ArrayRef) {
        let json = Arc::new(StringArray::from(vec![
            r#"{"k":1}"#,
            r#"{"k":2}"#,
            r#"{"k":3}"#,
        ])) as ArrayRef;
        let attributes = json_to_variant(&json).unwrap();
        let attributes_field = Arc::new(attributes.field("attributes"));
        let element = Fields::from(vec![
            Arc::new(Field::new("name", DataType::Utf8, true)),
            Arc::clone(&attributes_field),
        ]);
        let structs = StructArray::try_new(
            element.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                attributes.into(),
            ],
            None,
        )
        .unwrap();
        let element = Arc::new(Field::new("element", DataType::Struct(element), true));
        let list = ListArray::try_new(
            Arc::clone(&element),
            OffsetBuffer::from_lengths([2, 1]),
            Arc::new(structs),
            None,
        )
        .unwrap();
        (
            Arc::new(Field::new("events", DataType::List(element), true)),
            Arc::new(list),
        )
    }

    /// The JSON of every `attributes` value of an `events` array.
    fn attributes_json(array: &ArrayRef) -> Vec<Option<String>> {
        let structs = array.as_list::<i32>().values().as_struct();
        variant_to_json(structs.column_by_name("attributes").unwrap())
            .unwrap()
            .iter()
            .map(|json| json.map(str::to_owned))
            .collect()
    }

    /// A Variant inside a List of Structs is found by its name path, shreds
    /// in place, and unshreds back to the same values.
    #[test]
    fn nested_variants_shred_and_unshred_in_place() {
        let (field, array) = events();
        let paths = variant_field_paths(&Fields::from(vec![Arc::clone(&field)]));
        assert_eq!(paths, vec![vec![
            "events".to_owned(),
            "element".to_owned(),
            "attributes".to_owned()
        ]]);
        let shredding =
            DataType::Struct(Fields::from(vec![Field::new("k", DataType::Int64, true)]));
        let (shredded_field, shredded) =
            map_variant_field(&field, &array, &paths[0], &mut |variant| {
                let variant = VariantArray::try_new(variant.as_ref()).unwrap();
                Ok(shred_variant(&variant, &shredding).unwrap().into())
            })
            .unwrap();
        assert_ne!(shredded_field.data_type(), field.data_type());
        let (unshredded_field, unshredded) = unshred_variants(&shredded_field, &shredded).unwrap();
        assert_eq!(unshredded_field.data_type(), field.data_type());
        assert_eq!(attributes_json(&unshredded), attributes_json(&array));
    }

    /// A List of Structs read without some children is rebuilt at its full
    /// type, the read children intact and the others placeholders.
    #[test]
    fn lists_of_structs_are_rebuilt_by_name() {
        let (field, array) = events();
        let structs = array.as_list::<i32>().values().as_struct();
        let names = Fields::from(vec![Field::new("name", DataType::Utf8, true)]);
        let read = Arc::new(
            ListArray::try_new(
                Arc::new(Field::new("element", DataType::Struct(names.clone()), true)),
                array.as_list::<i32>().offsets().clone(),
                Arc::new(
                    StructArray::try_new(names, vec![Arc::clone(structs.column(0))], None).unwrap(),
                ),
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        let rebuilt = rebuild_by_name(&read, field.data_type()).unwrap();
        assert_eq!(rebuilt.data_type(), field.data_type());
        let rebuilt = rebuilt.as_list::<i32>().values().as_struct();
        assert_eq!(rebuilt.column(0).as_ref(), structs.column(0).as_ref());
        assert_eq!(rebuilt.column(1).data_type(), structs.column(1).data_type());
    }
}
