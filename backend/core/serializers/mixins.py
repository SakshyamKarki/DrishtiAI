
class DynamicFieldsMixin:
    """
    Usage:
        ?fields=id,name
        ?exclude=email
    """
    def __init__(self, *args, **kwargs):
        fields = kwargs.pop('fields', None)
        exclude = kwargs.pop('exclude', None)

        super().__init__(*args, **kwargs)

        request = self.context.get('request')

        if request:
            fields_param = request.query_params.get('fields')
            exclude_param = request.query_params.get('exclude')

            if fields_param:
                fields = fields_param.split(',')

            if exclude_param:
                exclude = exclude_param.split(',')

        if fields is not None:
            allowed = set(fields)
            existing = set(self.fields)
            for field_name in existing - allowed:
                self.fields.pop(field_name)

        if exclude is not None:
            for field_name in exclude:
                self.fields.pop(field_name, None)