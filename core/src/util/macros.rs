#[macro_export]
macro_rules! mutex_data {
    ($mutex: expr) => {
        $mutex.lock().unwrap()
    };
}
