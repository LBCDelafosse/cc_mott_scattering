README
------

The core program is contained in four files (`collision.py`, `data.py`, `com_cross_sections.py` and `functions.py`). The data used to generate the plots can be found in `Exp` and in `calibration_data.asc`.

The `entry_file.py` file contains the functions used to generate the plots in the report. All these functions are called in `if __name__ == "main"` at the end of the file, but the calls are written as commentaries. The user just has to de-comment the call for the function they want to use, and then run the program.