def slot(request, name, lower=True):
    slots_ = request['request']['intent']['slots']
    if name in slots_:
        resolved = slots_[name]['resolutions']['resolutionsPerAuthority']
        res = resolved[0]['values'][0]['value']['name']
    else:
        res = list(slots_.values())[0]['value']
    if lower: res = res.lower()
    return '' if res == '?' else res