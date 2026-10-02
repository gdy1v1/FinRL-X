# FinRL-X Data Integrity Audit

- As of: **2026-09-27**
- Rows: **22909**
- Unique tickers: **715**
- Datadate range: **2015-06-30 → 2026-03-31**
- Duplicate (ticker, datadate): **0**
- y_return == 0: **0**
- y_return formula mismatches: **0**

## Null counts

- ticker: 0
- datadate: 0
- tradedate: 237
- actual_tradedate: 237
- trade_price: 821
- y_return: 1532

## Unresolved price tickers

AABA, ABC, ADS, ADT, AGN, AMTM, ANTM, APC, ARG, ARNC, BLL, BXLT, CBS, CELG, CHK, CMA, COG, COL, CPGX, CSRA, DISCA, DISCK, DISH, DNB, DO, DOW, ENDP, FB, FBHS, FL, FLIR, FLT, FOXA, FRC, FTR, GPS, HCP, IR, JEC, JWN, LB, LLTC, LM, LVLT, MMC, MNK, MON, MRO, MXIM, MYL, NBL, NLOK, NLSN, OGN, PCL, PDCO, PEAK, PKI, POM, PX, PXD, RAI, RE, RHT, SE, SNI, SPLS, SRCL, STI, STJ, SYMC, TE, TIF, TWC, UTX, VAR, VIAC, VNT, WFM, WLTW, WRK, WYND, XL, XLNX

> Missing delisted/unavailable prices and future returns are intentionally left NULL; they are never coerced to zero.
