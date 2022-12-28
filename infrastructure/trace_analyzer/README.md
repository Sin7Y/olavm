# generate trace table in EXCEL file

input: the trace file in JSON format. `generate_table.py` will copy data from trace file to EXCEL file. So people can analyze trace from rows.

* Install `xlsxwriter` package

```shell=
pip install xlsxwriter
```

* How to use it

```shell=
python infrastructure/trace_analyzer/generate_table.py executor/memory_trace.json
```