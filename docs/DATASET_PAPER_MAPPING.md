# Dataset-to-Paper Mapping

| Paper Variable | Dataset Column | Available | Transformation | Notes |
|---|---|---|---|---|
| Product ID / ASIN | ASIN | YES | Direct | Some duplicate ASINs exist (different months). |
| Product title | Title | YES | Direct | Text data available for feature extraction. |
| Product image | Main Image URL | PARTIAL | Unavailable | Only URL is provided. Actual images are NOT in the repository. Do not scrape. |
| Customer reviews | N/A | NO | Unavailable | Only aggregate rating counts exist, no review text for sentiment extraction. |
| Rating | Rating Score | YES | Direct | |
| Rating count | Total Rating Count | YES | Parse to int | |
| Review count | Total Review Count | YES | Parse to int | |
| BSR | Main Category BSR Ranking | YES | Parse to int | Requires handling commas/non-numeric chars. |
| Price | Buybox Price (USD) | YES | Parse to float | Requires handling missing/invalid characters. |
| Discount | Discount | YES | Direct | E.g., 5, 10, 15... |
| Promotion level | Discount | YES | Direct | Represents the same concept. |
| Historical price | N/A | NO | Unavailable | No separate historical price, but `Last Monthly Sales` is present. |
| Monthly sales | Last Monthly Sales | YES | Direct | |
| Gross margin | Gross Margin Rate | YES | Parse to float | Strip '%' |
| Listing duration | Listing Time / Duration | YES | Direct | |
| Month index | Deal Month | YES | Parse to categorical | e.g. January -> 1 |
| Category | Main Category BSR Category | YES | Direct | 27 categories identified. |
| Target (Sales Growth) | Estimated Monthly Sales Growth Rate | YES | Parse to float | Strip '%' |

## Critical Modality Absences
1. **Images:** The dataset only contains `Main Image URL`. The actual image files are missing from the repository. Without scraping Amazon (which is explicitly forbidden by project rules), the Image modality cannot be extracted locally. This means the 32-D Image representation of the 121-D ItaNet state cannot be populated as intended.
2. **Customer Reviews:** The dataset contains aggregate rating scores and counts, but completely lacks the text of customer reviews. The base paper extracts sentiment from top-10 reviews. Without review text, the 10-D Customer Review Sentiment representation cannot be extracted.

### Exact Consequence for Paper Reproduction
The absence of these two key unstructured modalities (Image and Customer Reviews) prevents the full 121-D ItaNet state from being exactly constructed. The available modalities are only:
- Product Title (32-D)
- Structured Attributes (47-D)

We will proceed with preprocessing the available modalities and structured attributes, but the final extracted state will lack the image and review sentiment dimensions, or they must be represented as zeros/omitted as per the missing data policy. We will not hallucinate synthetic images or synthetic reviews.
