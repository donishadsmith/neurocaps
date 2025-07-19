{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
      :nosignatures:
      :template: classmethod.rst

      {% if objname != 'PlotDefaults' %}
      ~{{ objname }}.__str__
      {% endif %}


   {% for item in methods %}
      {%  if not item.startswith('_') %} ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endblock %}

   .. raw:: html

       <div style='clear:both'></div>
