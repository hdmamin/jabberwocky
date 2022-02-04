import logging

import ask_sdk_core.utils as ask_utils, is_request_type, is_intent_name
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestInterceptor
from ask_sdk_core.dispatch_components import AbstractResponseInterceptor
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_model.dialog import DelegateRequestDirective, \
    DelegationPeriodUntil

from jabberwocky.openai_utils import ConversationManager

import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):

    on_start_text = 'Who would you like to speak to?'

    def can_handle(self, handler_input):
        return is_request_type('LaunchRequest')(handler_input)

    def handle(self, handler_input):
        handler_input.response_builder.speak(self.on_start_text)
        return handler_input.response_builder.response


class ChoosePersonHander(AbstractRequestHandler):

    def can_handle(self):
        # TODO
        pass

    def handle(self):
        # TODO: consider how to make this persist (if possible? Sounds like
        # lamda is intended to be stateless to unsure how this works).
        # MANAGER = ConversationManager([person])
        pass


class ChooseModelHandler(AbstractRequestHandler):

    def can_handle(self, handler_input):
        return is_intent_name('chooseModel')(handler_input)

    def handle(self, handler_input):
        # TODO: change model using jabberwocky.
        model = handler_input.request_envelope.request.intent.slots\
            .get('Model')
        if model is None:
            handler_input.response.builder.speak(
                'I didn\'t understand that model type. You are still using '
                f'{MANAGER._kwargs["model_i"]}'
            )
        else:
            handler_input.response_builder.speak('You are now using model ' +
                                                 model + '.')
        return handler_input.response_builder.response


class RecordColorApiHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return util.is_api_request(handler_input, 'RecordColor')

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # First get our request entity and grab the color passed in the
        # API call.
        args = util.get_api_arguments(handler_input)
        color = args['color']

        # Store the favorite color in the session
        session_attributes = handler_input.attributes_manager.session_attributes
        session_attributes['favoriteColor'] = color

        response = {
            "apiResponse": {
                "color": color
            },
            "shouldEndSession": False
        }

        return response


class IntroToAlexaConversationsButtonEventHandler(AbstractRequestHandler):
    """Handler for buttons in the APL display"""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("Alexa.Presentation.APL.UserEvent")(handler_input) and handler_input.request_envelope.request.arguments[0] == 'SetFavoriteColor'

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        return handler_input.response_builder.add_directive(
            DelegateRequestDirective(
                target="AMAZON.Conversations",
                period={
                    "until": "EXPLICIT_RETURN"
                },
                updated_request={
                    "type": "Dialog.InputRequest",
                    "input": {
                        "name": "SpecifyFavoriteColor",
                        "slots": {
                            "color": {
                                "name": "color",
                                "value": handler_input.request_envelope.request.arguments[1]
                            }
                        }
                    }
                }
            )
        ).response


class GetFavoriteColorApiHandler(AbstractRequestHandler):
    """Retrieves the user's favorite color, if set"""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return util.is_api_request(handler_input, 'GetFavoriteColor')

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Check whether 'favoriteColor" key exists within sessionAttributes
        # object and if so set color
        session_attributes = handler_input.attributes_manager.session_attributes

        if 'favoriteColor' in session_attributes.keys():
            color = session_attributes['favoriteColor']
        else: color = None

        response = {
            "apiResponse": {
                "color" : color
            },
            "shouldEndSession": False
        }

        return response


class FallbackIntentHandler(AbstractRequestHandler):
    """Handler for Fallback Intent.
    AMAZON.FallbackIntent is only available in en-US locale.
    This handler will not be triggered except in that locale,
    so it is safe to deploy on any locale.
    """
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        logger.info("In FallbackIntentHandler")
        speech = "Hmm, I'm not sure. You can tell me your favorite color or ask me what your favorite color is. What would you like to do?"
        reprompt = "I didn't catch that. What can I help you with?"

        return handler_input.response_builder.speak(speech).ask(reprompt).response


class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End"""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "You just triggered " + intent_name + "."

        return (
            handler_input.response_builder
            .speak(speak_output)
            # .ask("add a reprompt if you want to keep the session open for the user to respond")
            .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """

    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class LoggingRequestInterceptor(AbstractRequestInterceptor):
    """Log the request envelope."""
    def process(self, handler_input):
        # type: (HandlerInput) -> None
        logger.info("Request Envelope: {}".format(
            handler_input.request_envelope))


class LoggingResponseInterceptor(AbstractResponseInterceptor):
    """Log the response envelope."""
    def process(self, handler_input, response):
        # type: (HandlerInput, Response) -> None
        logger.info("Response: {}".format(response))


# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.
sb = SkillBuilder()

# register request / intent handlers
sb.add_request_handler(RecordColorApiHandler())
sb.add_request_handler(GetFavoriteColorApiHandler())
sb.add_request_handler(IntroToAlexaConversationsButtonEventHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler())

# register exception handlers
sb.add_exception_handler(CatchAllExceptionHandler())

# register interceptors
sb.add_global_request_interceptor(LoggingRequestInterceptor())
sb.add_global_response_interceptor(LoggingResponseInterceptor())

lambda_handler = sb.lambda_handler()
