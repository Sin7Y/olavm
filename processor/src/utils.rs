use plonky2::field::goldilocks_field::GoldilocksField;
use std::os::raw::c_char;
use vm_core::program::Program;
use xlsxwriter::{Format, FormatAlignment, FormatColor, FormatUnderline, Workbook};

pub fn export_trace_table(file: &str, program: Program) {
    let workbook = Workbook::new(file);
}

#[test]
fn export_excel_test() {
    let workbook = Workbook::new("simple1.xlsx");

    let mut sheet1 = workbook.add_worksheet(None).unwrap();
    sheet1.write_string(0, 0, "Red text", None).unwrap();
    sheet1.write_number(0, 1, 20., None).unwrap();
    sheet1.write_formula_num(1, 0, "=10+B1", None, 30.).unwrap();
    sheet1
        .write_url(
            1,
            1,
            "https://github.com/informationsea/xlsxwriter-rs",
            None,
        )
        .unwrap();
    sheet1
        .merge_range(2, 0, 3, 2, "Hello, world", None)
        .unwrap();

    sheet1.set_selection(1, 0, 1, 2);
    sheet1.set_tab_color(FormatColor::Cyan);
    workbook.close();
}
