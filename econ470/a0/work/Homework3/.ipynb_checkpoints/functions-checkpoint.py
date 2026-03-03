"""
Helper functions for Medicare Advantage data processing.
Python equivalent of functions.R
"""
import pandas as pd
import numpy as np


def read_contract(path):
    """Read contract info CSV file."""
    col_names = [
        "contractid", "planid", "org_type", "plan_type", "partd", "snp", "eghp",
        "org_name", "org_marketing_name", "plan_name", "parent_org", "contract_date"
    ]
    df = pd.read_csv(path, skiprows=1, names=col_names, encoding="latin1", dtype={
        'contractid': str, 'planid': float, 'org_type': str, 'plan_type': str,
        'partd': str, 'snp': str, 'eghp': str, 'org_name': str,
        'org_marketing_name': str, 'plan_name': str, 'parent_org': str, 'contract_date': str
    })
    return df


def read_enroll(path):
    """Read enrollment info CSV file."""
    col_names = ["contractid", "planid", "ssa", "fips", "state", "county", "enrollment"]
    df = pd.read_csv(path, skiprows=1, names=col_names, na_values="*", encoding="latin1", dtype={
        'contractid': str, 'planid': float, 'ssa': float, 'fips': float,
        'state': str, 'county': str, 'enrollment': float
    })
    return df


def load_month(m, y):
    """Load one month of plan/enrollment data."""
    c_path = f"../../ma-data/ma/enrollment/Extracted Data/CPSC_Contract_Info_{y}_{m}.csv"
    e_path = f"../../ma-data/ma/enrollment/Extracted Data/CPSC_Enrollment_Info_{y}_{m}.csv"

    contract_info = read_contract(c_path).drop_duplicates(subset=['contractid', 'planid'], keep='first')
    enroll_info = read_enroll(e_path)

    merged = contract_info.merge(enroll_info, on=['contractid', 'planid'], how='left')
    merged['month'] = int(m)
    merged['year'] = y
    return merged


def read_service_area(path):
    """Read service area CSV file."""
    col_names = [
        "contractid", "org_name", "org_type", "plan_type", "partial", "eghp",
        "ssa", "fips", "county", "state", "notes"
    ]
    df = pd.read_csv(path, skiprows=1, names=col_names, na_values="*", encoding="latin1", dtype={
        'contractid': str, 'org_name': str, 'org_type': str, 'plan_type': str,
        'partial': str, 'eghp': str, 'ssa': float, 'fips': float,
        'county': str, 'state': str, 'notes': str
    })
    # Convert partial to boolean
    df['partial'] = df['partial'].map({'TRUE': True, 'FALSE': False, True: True, False: False})
    return df


def load_month_sa(m, y):
    """Load one month of service area data."""
    path = f"../../ma-data/ma/service-area/Extracted Data/MA_Cnty_SA_{y}_{m}.csv"
    df = read_service_area(path)
    df['month'] = int(m)
    df['year'] = y
    return df


def read_penetration(path):
    """Read penetration CSV file."""
    col_names = [
        "state", "county", "fips_state", "fips_cnty", "fips",
        "ssa_state", "ssa_cnty", "ssa", "eligibles", "enrolled", "penetration"
    ]
    df = pd.read_csv(path, skiprows=1, names=col_names, na_values=['', 'NA', '*', '-', '--'], encoding="latin1", dtype={
        'state': str, 'county': str, 'fips_state': 'Int64', 'fips_cnty': 'Int64',
        'fips': float, 'ssa_state': 'Int64', 'ssa_cnty': 'Int64', 'ssa': float,
        'eligibles': str, 'enrolled': str, 'penetration': str
    })
    # Parse numeric columns (handles commas)
    for col in ['eligibles', 'enrolled', 'penetration']:
        df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('%', ''), errors='coerce')
    return df


def load_month_pen(m, y):
    """Load one month of penetration data."""
    path = f"../../ma-data/ma/penetration/Extracted Data/State_County_Penetration_MA_{y}_{m}.csv"

    df = pd.read_csv(
        path,
        na_values=["UK", "NA", "*", "."],
        low_memory=False
    )

    df["month"] = int(m)
    df["year"] = y
    return df


def mapd_clean_merge(ma_data, mapd_data, y):
    """Clean and merge MA and MA-PD landscape data."""
    # Tidy MA-only data
    ma_data = ma_data[['contractid', 'planid', 'state', 'county', 'premium']].copy()
    
    ma_data["contractid"] = ma_data["contractid"].astype("string").str.upper().str.strip()
    ma_data["state"] = ma_data["state"].astype("string").str.strip()
    ma_data["county"] = ma_data["county"].astype("string").str.strip()
    ma_data["planid"] = pd.to_numeric(ma_data["planid"], errors="coerce").astype("Int64")
    
    ma_data["premium"] = (
        ma_data["premium"].astype("string")
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
    )
    ma_data["premium"] = pd.to_numeric(ma_data["premium"], errors="coerce")
    
    # Fill missing premium within groups
    ma_data = ma_data.sort_values(['contractid', 'planid', 'state', 'county'])
    
    fill_cols = ['premium_partc', 'premium_partd_basic', 'premium_partd_supp', 'premium_partd_total', 'partd_deductible']
    
    for c in fill_cols:
        mapd_data[c] = (
            mapd_data[c].astype("string")
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )
        # This line must be indented relative to the 'for' loop
        mapd_data[c] = pd.to_numeric(mapd_data[c], errors="coerce")

    ma_data['premium'] = ma_data.groupby(['contractid', 'planid', 'state', 'county'])['premium'].ffill()

    # Remove duplicates
    ma_data = ma_data.drop_duplicates(subset=['contractid', 'planid', 'state', 'county'], keep='first')

    # Tidy MA-PD data
    mapd_data = mapd_data[['contractid', 'planid', 'state', 'county', 'premium_partc',
                          'premium_partd_basic', 'premium_partd_supp', 'premium_partd_total',
                          'partd_deductible']].copy()
    
    mapd_data["planid"] = pd.to_numeric(mapd_data["planid"], errors="coerce").astype("Int64")

    # Fill missing values within groups
    mapd_data = mapd_data.sort_values(['contractid', 'planid', 'state', 'county'])
    mapd_data[fill_cols] = mapd_data.groupby(['contractid', 'planid', 'state', 'county'])[fill_cols].ffill()

    # Remove duplicates
    mapd_data = mapd_data.drop_duplicates(subset=['contractid', 'planid', 'state', 'county'], keep='first')

    # Merge Part D info to Part C info
    plan_premiums = ma_data.merge(mapd_data, on=['contractid', 'planid', 'state', 'county'], how='outer')
    plan_premiums['year'] = y

    return plan_premiums