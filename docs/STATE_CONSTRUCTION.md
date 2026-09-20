# State Construction

## Static vs Dynamic Features
The paper distinguishes between static context and dynamic signals:
- **Static Context ($c_i$):** Product identity, category, brand (extracted mostly through text embeddings of Title and categorical structured attributes).
- **Dynamic Signals ($d_{t,i}$):** Time-varying factors such as Price, Rating Counts, Sales, BSR Rankings, and Month.

**Dataset Reality:** The dataset has multiple observations for the same product across different months, providing a true dynamic signal for BSR, reviews count, and price.

## Modality Availability & Final State
The intended ItaNet state is 121-D:
- Image: 32-D (Unavailable)
- Text: 32-D (Available)
- Sentiment: 10-D (Unavailable)
- Attributes: 47-D (Available)

**Implementation Deviation:** Because Images and Customer Reviews are missing from the dataset (and fabricating them is prohibited), the state will consist only of available modalities: Text (32-D) and Attributes (47-D). 
**Final State Dimension:** 79-D.

## Feature Fusion
The fusion mechanism is a direct concatenation: `[Text, Attributes]`. 
No cross-attention or multimodal transformers are introduced, preserving the paper's original intent without hallucinating synthetic modalities.
