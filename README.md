# high dimensional unscented transformation
 
 Code for the paper "A Numerically Stable High-dimensional Unscented Transformation"

 The files to run to reproduce the results are:
 -scripts
    - main_column_A.py: To run simulations of the four-component distillation column with state estimators (Section 9 in the paper).
    - plotter_colA.py: Plot main results based on saved data from main_column_A.py.
    - y_equal_x2_independent_hd.py: Run toy example y=x^2 with independent x (Section 7.1 in the paper)
    - y_equal_x2_comparison_hdut_vs_mc.py: Run toy example y=x^2 with correlated x (Section 7.2 in the paper)

Other files which may be interesting to run:
-scripts
    - alpha_sut_to_make_Wc0_positive.py: Shows which value of alpha that makes W_c^0 negative
    - main_sigma_point_plot_sqrt.py: Shows the effect on the sigma-points in 2D when using different matrix square-roots.
    - illustration_of_hdut_using_normalized_rv.py: Implementing HD-UT as in Section 5.2 (not the optimal way of implementing it (not Algorithm 1), but shows the reasoning of the approach).

Remaining scripts contain utility functions. Data for the case studies are in
- data_y_x2_corr_x: data for y=x^2 when x is correlated
- data_colA: data for the case study in Section 9.

The code works using Python 3.11.5, CasADi 3.6.3, numpy xxx, scipy xxx, pandas xxx, matplotlib xxx, seaborn xxx

