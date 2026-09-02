Performance
===========

The emission fit can become slow when the absorption fit contains many CNM
components. Version 1.0.0 keeps the same radiative-transfer and BIC fitting
method while avoiding equivalent orderings of separated CNM components and
using analytic model derivatives.

For ``ncold`` cold components and ``nwarm`` warm components, the number of
candidate nonlinear emission fits can scale roughly as:

.. code-block:: text

   ncold! * len(F) ** nwarm

Search modes
------------

The default ``cnm_order_strategy="overlap"`` tests foreground/background
orders within groups of blended CNM components. Separated components do not
attenuate one another appreciably, so globally permuting them duplicates the
same model. The default also caps the automatic WNM search at six components
and stops as soon as adding a component does not improve the BIC selection
score.

For an exact legacy ordering audit, use:

.. code-block:: python

   spec_fit.cnm_order_strategy = "exhaustive"
   spec_fit.max_cnm_orderings = None

This compatibility mode can still be factorially expensive.

Measured complex-spectrum benchmark
-----------------------------------

Version 1.0.0 was tested on the 195-channel SMC spectrum
``J011049-731427.txt`` with automatic absorption and emission centers. The
absorption fit selected five CNM components. A 60-second run of version 0.1.1
timed out after 1,925 nonlinear emission fits and had not completed. Based on
the 32,520 candidates used by the corresponding exhaustive search, the measured
lower-bound estimate for version 0.1.1 is about 17 minutes on the benchmark
machine, with longer runtimes expected as higher-component fits become slower.
The new default search completed in about 13 seconds using 1,626 fits, selected
two WNM components, and obtained a weighted emission chi-square of 43.927.

For comparison, version 1.0.0 in exhaustive compatibility mode tested 9,840
fits (with the WNM search capped at the selected two components), completed in
about 45 seconds, selected the same component counts, and obtained a weighted
emission chi-square of 43.918. The small chi-square difference confirms that
discarded global orderings were effectively equivalent for this spectrum. The
largest fitted spin-temperature difference was 0.085 K, and the largest WNM
center and width differences were 0.012 and 0.010 km/s, respectively.

The legacy automatic loop could also continue indefinitely when its best BIC
remained above a hard-coded absolute threshold. Version 1.0.0 instead stops
after the first non-improving BIC component count, so this failure mode no
longer explains multi-day or multi-week runs.

The reusable benchmark script and generated plots are stored in ``test_code``
next to the package repository.

Practical speed tips
--------------------

- Set ``peak_emi`` manually when you know the likely emission component
  centers. This avoids a larger automatic search.
- Reduce ``F`` during exploration. For example, use ``spec_fit.F = [0.5]`` for
  a fast first pass, then restore ``[0, 0.5, 1]`` for a final run.
- Limit absorption components by setting ``peak_abs`` manually or using
  ``num_cold`` when the automatic fit over-splits noisy absorption spectra.
- Lower ``max_auto_warm_components`` from its default of ``6`` when you want a
  faster exploratory automatic fit.
- Keep ``align_data=True`` when absorption and emission velocity grids differ.

Example fast first pass
-----------------------

.. code-block:: python

   spec_fit.peak_emi = []
   spec_fit.max_auto_warm_components = 1
   spec_fit.F = [0.5]
   spec_fit.num_cold = 5
   spec_fit.fit_and_plot()

After the fit is stable, rerun with the fuller configuration if needed:

.. code-block:: python

   spec_fit.F = [0, 0.5, 1]
   spec_fit.num_cold = 0
   spec_fit.renew = True
   spec_fit.fit_and_plot()
