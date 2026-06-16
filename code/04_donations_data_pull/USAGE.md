# Summary of Steps 

## Per race
- downloads dwbexport.zip from CalAccess
- merges RCPT_CD (Schedule a/C) 
- classify committees to a race using name 
- filters for donation receipts > 2025-01-01 and target committees
- normalizes dates, extracts cycles, extracts transaction types
- deduplicates on a composite key (e.g., drops rows that have the same date, amount, contributor, committeee, and contribution type)

