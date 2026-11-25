# Problem-Set-5-Econ_3916_PS5
# Problem Set 5: The Vendor's 93%

ECON 3916 · Applied Data Analytics · Northeastern University

A vendor sold a lender a loan-default model on "93% accuracy", and the head of risk was ready to sign. But 93% of applications don't default, so a model that approves everyone is also right 93% of the time. This notebook simulates 20,000 loan applications from a known default process and builds the baseline the vendor didn't. It fits penalised logistic regression and a random forest, puts a price on each kind of error, and finds the threshold at which the lender should actually decline an application.

**Notebook:** [`Econ_3916_PS5.ipynb`](Econ_3916_PS5.ipynb)

## The recommendation to the head of risk

**Don't sign on the 93%.** On our test applications, a model that approves everyone scores 92.9%. A 93% accuracy is what a model that has learned nothing gets, so it proves nothing about the vendor's model.

**What a model is actually worth.** Our own five-field logistic model has the same accuracy (93.1%) but ranks risk well: PR AUC 0.32, against 0.07 for no information. Its value lies in whom it declines. At the threshold below, it cuts expected credit cost from USD 939 per application (approve everyone) to USD 593. That is 37% lower, about USD 3.5 million per 10,000 applications. A random forest did worse (USD 658), so complexity buys nothing here. Ask the vendor to show its cost on our data against this benchmark, and sign only if it beats USD 593.

**Threshold: decline when the predicted default probability exceeds 14% (1 in 7), not 50%.** This assumes a default loses 60% of principal and a wrongly declined good applicant costs 10% (8% margin plus 2% reputational). The rule is: cost of a wrong decline ÷ (that cost + loss on a default). At 50%, the model declines 1% of applicants and captures under a third of the saving. At 14%, it declines 13% and catches half of all defaults.

**What would change my answer:** the probabilities drifting. The 14% rule works because a predicted 14% currently means 14 defaults in 100. A downturn would break that, so recheck calibration against realised defaults every quarter.

## What it shows

All performance numbers are on a 6,000-application test set (425 defaults) that was split off before anything else was looked at.

| | Result |
|---|---|
| Null model ("nobody defaults") | 92.92% accuracy; catches 0 of 425 defaults |
| Plain logistic regression | all 5 real features significant (p < 10⁻²¹); 1 of 6 noise features (`form_fill_secs`) also passes p < 0.05 |
| LASSO, cross-validated penalty | zeros 1 of 6 noise features and 0 of 5 real ones; the stricter one-SE rule zeros 5 of 6 noise and still no real feature |
| Penalised logistic model | 93.13% accuracy, 13 more correct answers than the null out of 6,000; ROC AUC 0.823, PR AUC 0.321 |
| Random forest, 500 trees | out-of-bag accuracy 0.9303, test 0.9303; loses to the logistic model on every measure (ROC AUC 0.794, PR AUC 0.293) |
| Forest vs logistic, McNemar's test | the models disagree on 46 applicants, split 20 to 26; p = 0.46, indistinguishable on accuracy |
| Built-in (MDI) vs permutation importance | MDI gives the six noise features 38.6% of importance, ranked exactly by how many distinct values each takes; permutation importance puts every one within 0.003 AUC of zero |
| Importance is not effect | halving the loan for the riskiest tenth: the model predicts default falling 9.0 points, the truth is 5.9, because stressed applicants ask for bigger loans |
| Expected cost per application | approve all USD 939; threshold 0.5 USD 827; cost minimum (threshold 0.159) USD 585; rule 1/7 USD 593 |
| At the cost-minimising threshold | recall 10% → 50%, precision 59% → 31%, accuracy 93.1% → 88.7%, which is below the null |
| Optimal threshold vs cost ratio | tracks 1 / (1 + ratio): median gap 0.009 for ratios 3 to 30 |
| Calibration | predicted 7.04% vs observed 7.08% defaults, calibration slope 0.995; Platt and isotonic recalibration don't improve it, so the threshold holds |

## Structure

1. **Build the baseline nobody built.** 20,000 simulated applications at 7% default. Five real features (income, debt-to-income with a kink at 40%, credit history, prior delinquencies, loan amount), six pure-noise features, and a hidden "financial stress" variable no model sees. Train/test split first, then the null model.
2. **Fit and regularise.** Plain, LASSO and Ridge logistic regression with the penalty chosen by 5-fold cross-validation, checked against the known noise features. Then a random forest with out-of-bag error, built-in vs permutation importance, and the difference between a predictive feature and a lever the lender can pull.
3. **The number that actually matters.** Confusion matrix, precision, recall, F1, ROC and precision-recall curves. An expected-cost curve over every threshold, costed on each loan's principal, and how the optimal threshold moves with the cost ratio.
4. **AI expansion.** Two P.R.I.M.E. prompts sent to Claude Opus 5.5, each with something deliberately left out:
   - Part A, a significance test of forest vs logistic. It chose McNemar's test correctly but guessed the forest had won.
   - Part B, a calibration check. It was run with its three placeholder constants replaced.

   Both AI replies are reproduced unedited in the notebook.

## Running it

Open the notebook in Google Colab and choose **Runtime › Run all**. There is nothing to install and no data file: numpy, pandas, scipy, statsmodels, scikit-learn and matplotlib come preinstalled, and the applications are simulated in the notebook. The simulation uses one seeded generator (`numpy.random.default_rng(42)`) and every model uses `random_state=42`, so a rerun reproduces every number. A full run takes a few minutes; the random forest and its permutation importance are the slowest steps.
