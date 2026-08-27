from policy import normalize_label, validate_values


def build_record(label, values):
    cleaned = validate_values(values)
    return {
        "label": label,
        "total": sum(cleaned),
        "count": len(cleaned),
    }
