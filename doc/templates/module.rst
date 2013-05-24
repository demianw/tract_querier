{{ fullname }}
{{ underline }}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
  :template: class.rst
  {% for item in classes %}
    {{ fullname }}.{{ item }}
  {%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
  {% for item in functions %}
    {{ fullname }}.{{ item }}
  {%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
  {% for item in exceptions %}
    {{ fullname }}.{{ item }}
  {%- endfor %}
{% endif %}
{% endblock %}



.. automodule:: {{fullname}}
   :members:
   :undoc-members:
   :show-inheritance:




