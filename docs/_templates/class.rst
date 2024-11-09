:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
   {% for item in methods %}
      {% if item != "__init__" %} ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endblock %}

   .. raw:: html

       <div style='clear:both'></div>
