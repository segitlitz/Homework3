  read_contract <- function(path) {
    read_csv(
      path,
      skip = 1,
      col_names = c(
        "contractid","planid","org_type","plan_type","partd","snp","eghp",
        "org_name","org_marketing_name","plan_name","parent_org","contract_date"
      ),
      col_types = cols(
        contractid = col_character(),
        planid     = col_double(),
        org_type   = col_character(),
        plan_type  = col_character(),
        partd      = col_character(),
        snp        = col_character(),
        eghp       = col_character(),
        org_name   = col_character(),
        org_marketing_name = col_character(),
        plan_name  = col_character(),
        parent_org = col_character(),
        contract_date = col_character()
      ),
      show_col_types = FALSE,
      progress = FALSE
    )
  }

  read_enroll <- function(path) {
    read_csv(
      path,
      skip = 1,
      col_names = c("contractid","planid","ssa","fips","state","county","enrollment"),
      col_types = cols(
        contractid = col_character(),
        planid     = col_double(),
        ssa        = col_double(),
        fips       = col_double(),
        state      = col_character(),
        county     = col_character(),
        enrollment = col_double()
      ),
      na = "*",
      show_col_types = FALSE,
      progress = FALSE
    )
  }

  # One-month loader --------------------------------------------------------
  load_month <- function(m, y) {
    c_path <- paste0("../../ma-data/ma/enrollment/Extracted Data/CPSC_Contract_Info_", y, "_", m, ".csv")
    e_path <- paste0("../../ma-data/ma/enrollment/Extracted Data/CPSC_Enrollment_Info_", y, "_", m, ".csv")

    contract.info <- read_contract(c_path) %>%
      distinct(contractid, planid, .keep_all = TRUE)   

    enroll.info <- read_enroll(e_path)

    contract.info %>%
      left_join(enroll.info, by = c("contractid","planid")) %>%
      mutate(month = as.integer(m), year = y)
  }

  read_service_area <- function(path) {
    read_csv(
      path, skip = 1,
      col_names = c(
        "contractid","org_name","org_type","plan_type","partial","eghp",
        "ssa","fips","county","state","notes"
      ),
      col_types = cols(
        contractid = col_character(),
        org_name   = col_character(),
        org_type   = col_character(),
        plan_type  = col_character(),
        partial    = col_logical(),
        eghp       = col_character(),
        ssa        = col_double(),
        fips       = col_double(),
        county     = col_character(),
        state      = col_character(),
        notes      = col_character()
      ),
      na = "*",
      show_col_types = FALSE,
      progress = FALSE
    )
  }

  # One-month loader --------------------------------------------------------
  load_month_sa <- function(m, y) {
    path <- paste0("../../ma-data/ma/service-area/Extracted Data/MA_Cnty_SA_",y, "_", m, ".csv")
    
    read_service_area(path) %>%
      mutate(month = as.integer(m), year = y)
  }

mapd.clean.merge <- function(ma.data, mapd.data, y) {
  
  # Tidy MA-only data -------------------------------------------------------
  ma.data <- ma.data %>%
    select(contractid, planid, state, county, premium)
  
  ## Fill in missing plan info (by contract, plan, state, and county)
  ma.data <- ma.data %>%
    group_by(contractid, planid, state, county) %>%
    fill(premium)
  
  ## Remove duplicates
  ma.data <- ma.data %>%
    group_by(contractid, planid, state, county) %>%
    mutate(id_count=row_number())
  
  ma.data <- ma.data %>%
    filter(id_count==1) %>%
    select(-id_count)
  
  
  # Tidy MA-PD data ---------------------------------------------------------
  mapd.data <- mapd.data %>% 
    select(contractid, planid, state, county, premium_partc, premium_partd_basic, 
           premium_partd_supp, premium_partd_total, partd_deductible) %>%
    mutate(planid=as.numeric(planid))
  
  mapd.data <- mapd.data %>%
    group_by(contractid, planid, state, county) %>%
    fill(premium_partc, premium_partd_basic, premium_partd_supp, premium_partd_total, partd_deductible)
  
  ## Remove duplicates
  mapd.data <- mapd.data %>%
    group_by(contractid, planid, state, county) %>%
    mutate(id_count=row_number())
  
  mapd.data <- mapd.data %>%
    filter(id_count==1) %>%
    select(-id_count)
  
  ## Merge Part D info to Part C info
  plan.premiums <- ma.data %>%
    full_join(mapd.data, by=c("contractid", "planid", "state", "county")) %>%
    mutate(year=y)
  
  return(plan.premiums)  
}

  read_penetration <- function(path) {
    raw <- read_csv(
      path,
      skip = 1,
      col_names = c(
        "state","county","fips_state","fips_cnty","fips",
        "ssa_state","ssa_cnty","ssa","eligibles","enrolled","penetration"
      ),
      # read potential problem columns as character first
      col_types = cols(
        state      = col_character(),
        county     = col_character(),
        fips_state = col_integer(),
        fips_cnty  = col_integer(),
        fips       = col_double(),
        ssa_state  = col_integer(),
        ssa_cnty   = col_integer(),
        ssa        = col_double(),
        eligibles  = col_character(),
        enrolled   = col_character(),
        penetration= col_character()
      ),
      na = c("", "NA", "*", "-", "--"),
      show_col_types = FALSE,
      progress = FALSE
    )

    # robust numeric parsing (handles commas, %, stray text)
    raw %>%
      mutate(
        eligibles   = parse_number(eligibles),
        enrolled    = parse_number(enrolled),
        penetration = parse_number(penetration)
      )
  }

  # One-month loader --------------------------------------------------------
  load_month_pen <- function(m, y) {
    path <- paste0("../../ma-data/ma/penetration/Extracted Data/State_County_Penetration_MA_",y, "_", m, ".csv")

    read_penetration(path) %>%
      mutate(month = as.integer(m), year = y)
  }


