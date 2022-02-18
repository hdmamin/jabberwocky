def slot(request, name, lower=True):
    resolved = request['request']['intent']['slots'][name]['resolutions']
    res = resolved['resolutionsPerAuthority'][0]['values'][0]['value']['name']
    if lower: res = res.lower()
    return res