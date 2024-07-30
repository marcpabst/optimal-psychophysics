use std::sync::Arc;

use arrow::{
    array::{Array, ArrayBuilder, FixedSizeListBuilder, PrimitiveBuilder},
    datatypes::Float64Type,
};

use nuts_rs::DrawStorage;

pub struct ArrowStorage {
    draws: FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>,
}

impl ArrowStorage {
    pub fn new(size: usize) -> ArrowStorage {
        let values = PrimitiveBuilder::new();
        let draws = FixedSizeListBuilder::new(values, size as i32);
        ArrowStorage { draws }
    }
}

impl DrawStorage for ArrowStorage {
    fn append_value(&mut self, point: &[f64]) -> anyhow::Result<()> {
        self.draws.values().append_slice(point);
        self.draws.append(true);
        Ok(())
    }

    fn finalize(mut self) -> anyhow::Result<Arc<dyn Array>> {
        Ok(ArrayBuilder::finish(&mut self.draws))
    }

    fn inspect(&self) -> anyhow::Result<Arc<dyn Array>> {
        Ok(ArrayBuilder::finish_cloned(&self.draws))
    }
}
