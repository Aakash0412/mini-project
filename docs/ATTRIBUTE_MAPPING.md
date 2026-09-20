# Attribute Mapping

The 47-D numerical/categorical attribute state is constructed from available structured fields. To achieve exactly 47 dimensions without arbitrary projections, we apply specific encodings:

| Paper Attribute Equivalent | Dataset Column | Processing | Output Dimension |
|---|---|---|---|
| Month | `MonthNum` | Numeric scaling | 1 |
| Total Rating | `Total Rating Count` | Numeric (fill NA w/ 0, scale) | 1 |
| New Rating | `New Rating Count` | Numeric (fill NA w/ 0, scale) | 1 |
| Total Reviews | `Total Review Count` | Numeric (fill NA w/ 0, scale) | 1 |
| New Reviews | `New Review Count` | Numeric (fill NA w/ 0, scale) | 1 |
| Rating Score | `Rating Score` | Numeric (fill NA w/ mean, scale)| 1 |
| Positive Rating Rate | `Positive Rating Rate` | Numeric (fill NA w/ 0) | 1 |
| Negative Rating Rate | `Negative Rating Rate` | Numeric (fill NA w/ 0) | 1 |
| Price | `Buybox Price (USD)` | Numeric (scale) | 1 |
| Weight | `Weight (pounds)` | Numeric (fill NA w/ mean, scale) | 1 |
| Volume | `Volume (cubic inches)` | Numeric (fill NA w/ mean, scale) | 1 |
| Variants | `Number of Variants` | Numeric (scale) | 1 |
| Number of Sellers | `Number of Sellers` | Numeric (scale) | 1 |
| FBA Shipping Fee | `FBA Shipping Fee (USD)`| Numeric (fill NA w/ mean, scale) | 1 |
| Gross Margin | `Gross Margin Rate` | Numeric (scale) | 1 |
| BSR Ranking | `Main Category BSR Ranking`| Numeric (fill NA w/ max, scale) | 1 |
| BSR Change | `Main Category BSR Change Number` | Numeric (fill NA w/ 0, scale) | 1 |
| BSR Change Rate | `Main Category BSR Change Rate` | Numeric (fill NA w/ 0, scale) | 1 |
| Category | `Main Category BSR Category`| One-Hot Encoding | 27 |
| Buybox Seller Type | `Buybox Seller Type` | One-Hot Encoding | 2 |

**Total Dimensions: 18 numeric + 27 category one-hot + 2 seller type one-hot = 47-D.**
