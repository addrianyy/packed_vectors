pub trait VectorConvertInto<T> {
    fn convert_vector(self) -> T;
}

pub trait VectorTransmuteInto<T> {
    fn transmute_vector(self) -> T;
}
