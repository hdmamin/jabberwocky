from ask_sdk_core.handler_input import HandlerInput

import ask_sdk_core.utils as ask_utils
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_api_request(handler_input, api_name):
    """Helper method to find if a request is for a certain apiName."""
    try:
        return (
            ask_utils.request_util.get_request_type(
                    handler_input) == 'Dialog.API.Invoked' and
            handler_input.request_envelope.request.api_request.name == api_name
        )
    except Exception as ex:
        logging.error(ex)
        return False


def get_api_arguments(handler_input):
    """Helper method to get API arguments from the request envelope."""
    try:
        return handler_input.request_envelope.request.api_request.arguments
    except Exception as ex:
        logging.error(ex)
        return False


def slot(handler_input, name):
    """Convenience function to extract slot value from a handler input. Unclear
    if this is better/worse than `get_api_arguments` which I didn't see til
    after writing this.

    Parameters
    ----------
    handler_input
    name: str
        Name of slot defined in alexa console, e.g. 'Person'.

    Returns
    -------
    str or None
    """
    return handler_input.request_envelope.request.intent.slots.get(name)


def respond(handler_input, text):
    handler_input.response_builder.speak(text)
    return handler_input.response_builder.response
