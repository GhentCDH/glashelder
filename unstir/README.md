# unstir

Match glass plates to reference photos of the same building, with DINOv2
retrieval, ALIKED + LightGlue and a fundamental matrix check.

- `glashelder unstir match`: match plates against the references into a match store
- `glashelder unstir decide`: decide again from the stored scores
- `glashelder unstir sample` and `serve`: make and label rounds of pairs
- `glashelder unstir lookup`: a web page that finds the building in an uploaded photo
