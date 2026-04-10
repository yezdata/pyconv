from datetime import datetime, timezone

def convert_ms_timestamp(timestamp_ms: float) -> str:
    """
    Převede Unix timestamp v milisekundách na formátovaný UTC řetězec.
    """
    if not isinstance(timestamp_ms, (int, float)):
        return "Chyba: Vstup musí být numerického typu."
    
    try:
        # Konverze milisekund na sekundy a vytvoření 'aware' objektu v UTC
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return dt.strftime('%H:%M:%S.%f')[:-3]
    except (ValueError, OSError, OverflowError) as e:
        return f"Chyba: Neplatný rozsah nebo formát ({e})"
