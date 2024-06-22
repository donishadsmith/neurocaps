Tutorials
=========
Tutorials for all classes and functions in neurocaps. **Note**, most of these examples use randomized data.
Output from the `Extracting Timeseries` section is from a test from Github Actions using a truncated version of an open dataset provided by `Laumann & Poldrack <https://openfmri.org/dataset/ds000031/>`_ 
and used in `Laumann et al., 2015 <https://doi.org/10.1016/j.neuron.2015.06.037>`_ [1]_ was also utilized. This data was obtained from the OpenfMRI database, accession number ds000031.


.. toctree::
   :hidden:

   extraction
   analysis
   merge
   standardize
   dtype

.. raw:: html

   <div class="sphx-glr-thumbnails">
      <div class="sphx-glr-thumbcontainer" tooltip="Tutorials for extracting (using parcellations), cleaning, pickling, and saving timeseries.">
      <a href="extraction.html">
   
.. only:: html

   .. image:: embed/thumbnail/visualize_timeseries_nodes.png
   
.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tutorial 1: Using <code>TimeseriesExtractor</code> For Spatial Dimensionality Reduction With Parcellations</div>
      </a>
      </div>



.. raw:: html

      <div class="sphx-glr-thumbcontainer" tooltip="Tutorials for performing Co-Activation Patterns (CAPs) and visualizing CAPs using multiple visualization methods.">
      <a href="analysis.html">
   
.. only:: html

   .. image:: embed/thumbnail/All_Subjects_CAP-1_radar.png
   
.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tutorial 2: Using <code>CAP</code> Within Groups or All Subjects</div>
      </a>
      </div>



.. raw:: html

      <div class="sphx-glr-thumbcontainer" tooltip="Tutorial to merge subject timeseries.">
      <a href="merge.html">
   
.. only:: html

   .. image:: embed/thumbnail/neurocaps.png
   
.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tutorial 3: Merging Timeseries with <code>merge_dicts</code></div>
      </a>
      </div>


.. raw:: html

      <div class="sphx-glr-thumbcontainer" tooltip="Tutorial to standardize timeseries within runs outside TimeseriesExtractor.">
      <a href="standardize.html">
   
.. only:: html

   .. image:: embed/thumbnail/neurocaps.png
   
.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tutorial 4: Standardizing Within Runs Using <code>standardize</code></div>
      </a>
      </div>



.. raw:: html

      <div class="sphx-glr-thumbcontainer" tooltip="Tutorial to change the dtype of each subject's numpy array to reduce memory usage.">
      <a href="dtype.html">
   
.. only:: html

   .. image:: embed/thumbnail/neurocaps.png
   
.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tutorial 5: Changing dtype with <code>change_dtype</code></div>
      </a>
      </div>

.. raw:: html

    </div>

==========

.. [1] Laumann, T. O., Gordon, E. M., Adeyemo, B., Snyder, A. Z., Joo, S. J., Chen, M. Y., Gilmore, A. W., McDermott, K. B., Nelson, S. M., Dosenbach, N. U., Schlaggar, B. L., Mumford, J. A., Poldrack, R. A., & Petersen, S. E. (2015). Functional system and areal organization of a highly sampled individual human brain. Neuron, 87(3), 657–670. https://doi.org/10.1016/j.neuron.2015.06.037