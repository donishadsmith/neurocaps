:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
      :nosignatures:

      ~{{ objname }}.__str__

   {% for item in methods %}
      {%  if not item.startswith('_') %} ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endblock %}

   .. raw:: html

       <div style='clear:both'></div>
