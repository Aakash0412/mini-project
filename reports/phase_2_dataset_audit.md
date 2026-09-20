# Dataset Audit Report

- **Number of rows:** 34041
- **Number of columns:** 38
- **Duplicate rows:** 0
- **Unique products (ASIN):** 22514
- **Duplicate product identifiers:** 11527
- **Unique categories (Main Category BSR Category):** 27
- **Date/time range (Listing Time):** ['2001-10-02 00:00:00', '2023-09-26 00:00:00']
- **Price range (Buybox Price (USD)):** None
- **Discount range (Discount):** [5.0, 40.0]

## Columns Summary
| Column | Type | Missing % | Unique values | Example values |
|---|---|---|---|---|
| Country Site | str | 0.0% | 1 | US, US, US |
| ASIN | str | 0.0% | 22514 | B08Z2SG9FC, B097RFPPYT, B08P2STS1R |
| Title | str | 0.0% | 22916 | ORSEN Colorful 8.5 Inch LCD Writing Tablet for Kids, Electronic Sketch Drawing Pad Doodle Board, Toddler Travel Learning Educational Toys Activity Games Birthday Gifts for 2 3 4 5 6 7 8 Year Old Girls, 7AM2M Sonic Electric Toothbrush with 6 Brush Heads for Adults and Kids, One Charge for 90 Days, Wireless Fast Charge, 5 Modes with 2 Minutes Build in Smart Timer, Electric Toothbrushes(Black), QEEIG Floating Shelves Wall Shelf 24 inches Long Farmhouse Bathroom Decor Bedroom Kitchen Living Room Wall Mounted 24 x 9 inch Set of 2, Rustic Brown (008-60BN) |
| PASIN | str | 0.0% | 21971 | B09SV6XP5K, B08MDGX88R, B091SVKXF2 |
| Listing Time | str | 0.0% | 2886 | 2020/6/24, 2020/11/4, 2021/1/18 |
| Listing Duration | int64 | 0.0% | 2947 | 947, 817, 743 |
| Main Category BSR Category | str | 0.0% | 27 | Toys & Games, Health & Household, Tools & Home Improvement |
| Main Category BSR Ranking | str | 0.0% | 18912 | 74, 648, 229 |
| Main Category BSR Change Number | str | 5.51% | 15598 | 64, -7, -84 |
| Main Category BSR Change Rate | str | 5.51% | 14186 | 640%, -1.07%, -26.84% |
| Subcategory BSR Category | str | 1.95% | 3480 | Doodle & Scribbler Boards, Sonic, Bathroom Shelves |
| Subcategory BSR Ranking | str | 1.95% | 1174 | 3, 2, 1 |
| Subcategory BSR Change Number | str | 6.35% | 1151 | 2, 0, 0 |
| Subcategory BSR Change Rate | str | 6.35% | 8265 | 200%, 0%, 0% |
| Buybox Price (USD) | str | 0.0% | 3259 | 13.98, 25.99, 39.82 |
| Total Rating Count | str | 0.0% | 8206 | 8,342, 16,176, 2,657 |
| New Rating Count | str | 0.0% | 1769 | 915, 893, 193 |
| Rating Score | float64 | 0.02% | 26 | 4.3, 4.4, 4.5 |
| Positive Rating Rate | str | 0.0% | 67 | 81%, 84%, 87% |
| Negative Rating Rate | str | 0.0% | 53 | 10%, 10%, 6% |
| Total Review Count | str | 0.0% | 3403 | 1,667, 2,534, 291 |
| New Review Count | str | 0.0% | 913 | 147, 156, 34 |
| Weight (pounds) | float64 | 16.2% | 1693 | 0.31, 0.53, 7.04 |
| Volume (cubic inches) | float64 | 20.99% | 12491 | 22.68, 10.0, 329.22 |
| Brand | str | 0.0% | 12485 | ORSEN, 7AM2M, QEEIG |
| Number of Variants | int64 | 0.0% | 347 | 3, 9, 13 |
| Buybox Seller | str | 0.0% | 13195 | DINFLY, 7AM2M, QEEIG |
| Buybox Seller Type | str | 0.0% | 2 | Third-party Seller, Third-party Seller, Third-party Seller |
| Buybox Seller Location | str | 0.0% | 49 | CN, CN, CN |
| Delivery Method | str | 0.0% | 3 | FBA, FBA, FBA |
| Number of Sellers | int64 | 0.0% | 52 | 2, 1, 6 |
| FBA Shipping Fee (USD) | float64 | 4.43% | 230 | 3.4, 4.24, 13.09 |
| Gross Margin Rate | str | 4.43% | 3425 | 60.66%, 68.68%, 39.50% |
| Main Image URL | str | 0.0% | 22260 | https://m.media-amazon.com/images/I/71UCOhz5esL._AC_SL1500_.jpg, https://m.media-amazon.com/images/I/71LC4l63UML._AC_SL1500_.jpg, https://m.media-amazon.com/images/I/81HgfwNj1YL._AC_SL1500_.jpg |
| Deal Month | str | 0.0% | 10 | January, January, January |
| Last Monthly Sales | int64 | 0.0% | 7257 | 59834, 62272, 13391 |
| Discount | int64 | 0.0% | 8 | 5, 5, 5 |
| Estimated Monthly Sales Growth Rate | str | 0.0% | 13343 | 18.89%, -57.74%, 95.85% |
