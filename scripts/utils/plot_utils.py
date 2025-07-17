def truncate(f, n):
    factor = 10.0**n
    return int(f * factor) / factor

def mm_to_inches(mm):
    return mm / 25.4

def m_to_um(res_in_m):
    return truncate(res_in_m * 1e6, 4)
