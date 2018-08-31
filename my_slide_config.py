# `my_slide_config.py` 
import os
print('config')
c = get_config()
c.TagRemovePreprocessor.remove_input_tags.add("to_remove")
c.SlidesExporter.reveal_theme="simple"


app_settings = {
    "postprocessor_class": "nbconvert.postprocessors.serve.ServePostProcessor",
    "export_format": "slides"
}
c.NbConvertApp.update(app_settings)

# the following does the equivalent of --no-prompt, see here: https://github.com/jupyter/nbconvert/blob/master/nbconvert/nbconvertapp.py#L109-L114
exporter_settings = {
    'exclude_input_prompt' : True,
    'exclude_output_prompt' : True,
}
c.TemplateExporter.update(exporter_settings)