# running tests
create `all_csv_files` folder and place there your datasets
then
```
from main import run_all_tests

run_all_tests()
```

# visualization
Open `visualized.ipynb` notebook and run it. Last cell will contain visualizations for all tests

# Questions?
1. What if one metric show positive results but another shows negative? - Another approach would be to just make decision via test which returned lowest p-value
2. How to you handle experiments with very low conversion rates? Either keep running the experiment, or perform additional data analysis to find what generates more value. 
3. What if the sample sizes are unbalanced? - One solution is to switch to other test. My implemented tests are not sensitive for unbalanced sample sizes.