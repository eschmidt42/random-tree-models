def expected_failed_checks(check) -> dict[str, str]:
    return {
        "check_do_not_raise_errors_in_init_or_set_params": "measure_name and similar parameters are Enums, this seems to not be expected by sklearn.",
        "check_parameters_default_constructible": "measure_name and similar parameters are Enums, this seems to not be expected by sklearn.",
        "check_estimators_nan_inf": "NaN and Inf are fine.",
    }
