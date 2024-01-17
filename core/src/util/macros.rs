#[macro_export]
macro_rules! mutex_data {
    ($mutex: expr) => {
        $mutex.lock().expect("locking mutex failed")
    };
}
