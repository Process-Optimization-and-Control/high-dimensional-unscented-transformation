# high dimensional unscented transformation
 
 Code for the paper "A Numerically Stable Unscented Transformation with Optimal Tuning Parameters for High-Dimensional Systems"

 The files to run to reproduce the results are:
 - scripts
 - coordinate_system_example_x_uniform.py: To run simulations of the coordinate transformation example, Section 7 in the paper.
 - y_equal_x2_independent_hd.py: Run toy example y=x^2 with independent x (Section 6.1 in the paper)
 - y_equal_x2_comparison_hdut_vs_mc.py: Run toy example y=x^2 with correlated x (Section 6.2 in the paper)

Other files which may be interesting to run:
- alpha_sut_to_make_Wc0_positive.py: Shows which value of alpha that makes W_c^0 negative
- main_sigma_point_plot_sqrt.py: Shows the effect on the sigma-points in 2D when using different matrix square-roots.
- illustration_of_hdut_using_normalized_rv.py: Implementing HD-UT as in Section 4.2 (not the optimal way of implementing it (not Algorithm 1), but shows the reasoning of the approach).

The code works using Python 3.11.5, CasADi 3.6.3, numpy 1.25.2, scipy 1.11.3, pandas 2.1.1, matplotlib 3.7.2, seaborn 0.13.0

