from pydantic import validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package, Image, Inputs, Outputs, Configs, Response, Request, Output, Input, Config
)


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, list):
            return "list"
        return "object"

    class Config:
        title = "Image"


class OutputText(Output):
    name: Literal["output"] = "output"
    value: Optional[str]
    type: Literal["string"] = "string"

    class Config:
        title = "Output"


class Classes(Output):
    name: Literal["classes"] = "classes"
    value: Union[List[str], str]
    type: str = "object"

    @validator("type", always=True)
    def set_type_based_on_value(cls, val, values):
        val = values.get("value")
        if isinstance(val, list):
            return "list"
        return "object"

    class Config:
        title = "Classes"


class InputClasses(Config):
    """
    Enter the list of classes as a JSON array.
    Example: ["cat", "dog", "bird"]
    Used for Classification, Multi-Label, and Object Detection tasks.
    """
    name: Literal["inputClasses"] = "inputClasses"
    value: List[str]
    type: Literal["list"] = "list"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Classes"
        json_schema_extra = {"shortDescription": "Class List"}


class InputPrompt(Config):
    """
    The custom prompt sent with the image to guide the model's response.
    Use this to specify the task, output format, or focus area.
    Leave blank to use the default task-specific prompt.
    """
    name: Literal["inputPrompt"] = "inputPrompt"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Prompt"
        json_schema_extra = {"shortDescription": "User Prompt"}


class InputAnthropicApiKey(Config):
    """
    Your Anthropic API key used to authenticate API requests.
    Obtain this from console.anthropic.com under API Keys.
    Keep it private and never expose it in client-side code.
    """
    name: Literal["inputApiKey"] = "inputApiKey"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Anthropic API Key"
        json_schema_extra = {"shortDescription": "sk-ant-..."}


class InputNovaVisionApiKey(Config):
    """
    Your NovaVision access token used to authenticate API requests.
    Obtain this from the NovaVision platform dashboard.
    Keep it private and never expose it in client-side code.
    """
    name: Literal["inputApiKey"] = "inputApiKey"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "NovaVision Access Token"
        json_schema_extra = {"shortDescription": "NovaVision Token"}


class AnthropicAPIConfigs(Configs):
    inputApiKey: InputAnthropicApiKey


class AnthropicAPIOption(Config):
    """
    Authenticate using your Anthropic API key.
    Select this option if you have a credited Anthropic account.
    Requires a valid key from console.anthropic.com.
    """
    name: Literal["Anthropic"] = "Anthropic"
    value: AnthropicAPIConfigs
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Anthropic API"
        json_schema_extra = {"target": "value", "shortDescription": "Anthropic API Key"}


class NovaVisionAPIConfigs(Configs):
    inputApiKey: InputNovaVisionApiKey


class NovaVisionOption(Config):
    """
    Authenticate using your NovaVision access token.
    Select this option if you are using the NovaVision platform.
    Requires a valid token from the NovaVision dashboard.
    """
    name: Literal["NovaVision"] = "NovaVision"
    value: NovaVisionAPIConfigs
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "NovaVision"
        json_schema_extra = {"target": "value", "shortDescription": "NovaVision Token"}


class APIProvider(Config):
    """
    Select Anthropic API if you have a credited Anthropic API key,
    or use the NovaVision access token by selecting NovaVision.
    """
    name: Literal["apiProvider"] = "apiProvider"
    value: Union[AnthropicAPIOption, NovaVisionOption]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "API Provider"
        json_schema_extra = {"shortDescription": "Anthropic or NovaVision"}


class VersionOpus46(Config):
    """
    Claude Opus 4.6 — the most capable model in the Claude 4 family.
    Best for complex reasoning and nuanced visual understanding tasks.
    Higher cost and latency compared to Sonnet and Haiku variants.
    """
    name: Literal["claude-opus-4-6"] = "claude-opus-4-6"
    value: Literal["claude-opus-4-6"] = "claude-opus-4-6"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Opus 4.6"
        json_schema_extra = {"shortDescription": "Most capable, highest cost"}


class VersionSonnet46(Config):
    """
    Claude Sonnet 4.6 — high performance with balanced speed and cost.
    Recommended for most production use cases.
    Good balance between capability, speed, and cost efficiency.
    """
    name: Literal["claude-sonnet-4-6"] = "claude-sonnet-4-6"
    value: Literal["claude-sonnet-4-6"] = "claude-sonnet-4-6"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Sonnet 4.6"
        json_schema_extra = {"shortDescription": "Balanced performance"}


class VersionSonnet45(Config):
    """
    Claude Sonnet 4.5 — a previous generation Sonnet model.
    Reliable performance for standard vision tasks.
    Slightly lower capability than Sonnet 4.6.
    """
    name: Literal["claude-sonnet-4-5"] = "claude-sonnet-4-5"
    value: Literal["claude-sonnet-4-5"] = "claude-sonnet-4-5"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Sonnet 4.5"
        json_schema_extra = {"shortDescription": "Previous Sonnet generation"}


class VersionHaiku45(Config):
    """
    Claude Haiku 4.5 — the fastest and most cost-effective model.
    Ideal for high-volume, latency-sensitive workloads.
    Lower capability than Sonnet or Opus models.
    """
    name: Literal["claude-haiku-4-5"] = "claude-haiku-4-5"
    value: Literal["claude-haiku-4-5"] = "claude-haiku-4-5"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Haiku 4.5"
        json_schema_extra = {"shortDescription": "Fastest, lowest cost"}


class VersionOpus45(Config):
    """
    Claude Opus 4.5 — high capability from the previous Opus generation.
    Suitable for complex tasks requiring deep visual understanding.
    Compare with Opus 4.6 to evaluate performance differences.
    """
    name: Literal["claude-opus-4-5"] = "claude-opus-4-5"
    value: Literal["claude-opus-4-5"] = "claude-opus-4-5"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Opus 4.5"
        json_schema_extra = {"shortDescription": "Previous Opus generation"}


class VersionSonnet4(Config):
    """
    Claude Sonnet 4 — the base Sonnet model from the Claude 4 generation.
    Suitable for general-purpose vision and language tasks.
    A balanced option for standard production workflows.
    """
    name: Literal["claude-sonnet-4"] = "claude-sonnet-4"
    value: Literal["claude-sonnet-4"] = "claude-sonnet-4"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Sonnet 4"
        json_schema_extra = {"shortDescription": "Claude 4 base model"}


class VersionOpus41(Config):
    """
    Claude Opus 4.1 — a high-capability model from the Opus 4.x line.
    Good choice for demanding multi-step reasoning and vision tasks.
    Intermediate between Opus 4 and Opus 4.5 in capability.
    """
    name: Literal["claude-opus-4-1"] = "claude-opus-4-1"
    value: Literal["claude-opus-4-1"] = "claude-opus-4-1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Opus 4.1"
        json_schema_extra = {"shortDescription": "High capability variant"}


class VersionOpus4(Config):
    """
    Claude Opus 4 — the foundational Opus 4 generation model.
    High capability for complex vision and reasoning tasks.
    Baseline Opus 4 performance without later incremental improvements.
    """
    name: Literal["claude-opus-4"] = "claude-opus-4"
    value: Literal["claude-opus-4"] = "claude-opus-4"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Opus 4"
        json_schema_extra = {"shortDescription": "Base Opus 4 model"}


class InputModelVersion(Config):
    """
    Select the Claude model version to use.
    Opus 4.6 is the most capable. Haiku 4.5 is fastest and most cost-effective.
    Sonnet models balance speed and intelligence.
    """
    name: Literal["inputModelVersion"] = "inputModelVersion"
    value: Union[VersionOpus46, VersionSonnet46, VersionSonnet45, VersionHaiku45, VersionOpus45, VersionSonnet4, VersionOpus41, VersionOpus4]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Model Version"
        json_schema_extra = {"shortDescription": "Claude Model"}


class ExtendedThinkingTrue(Config):
    """
    Enable extended thinking mode for deeper internal reasoning.
    The model reasons step-by-step before generating a response.
    Increases latency and token cost; temperature must be set to 1.
    """
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"
        json_schema_extra = {"shortDescription": "Enable deep reasoning"}


class ExtendedThinkingFalse(Config):
    """
    Disable extended thinking for standard response generation.
    The model responds without additional internal reasoning steps.
    Faster responses with normal token usage.
    """
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"
        json_schema_extra = {"shortDescription": "Standard processing"}


class ExtendedThinking(Config):
    """
    Enable Claude's extended thinking for deeper reasoning on complex tasks.
    When enabled, temperature cannot be used.
    Increases latency and cost but improves accuracy on difficult tasks.
    """
    name: Literal["extendedThinking"] = "extendedThinking"
    value: Union[ExtendedThinkingFalse, ExtendedThinkingTrue]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Extended Thinking"
        json_schema_extra = {"shortDescription": "Deep Reasoning"}


class ThinkingBudgetTokens(Config):
    """
    Maximum number of tokens for internal thinking when extended thinking is enabled.
    Higher values allow deeper reasoning but increase latency and cost.
    Minimum: 1024. Must be less than Max Tokens.
    """
    name: Literal["thinkingBudgetTokens"] = "thinkingBudgetTokens"
    value: int = 1024
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Thinking Budget Tokens"
        json_schema_extra = {"shortDescription": "Thinking Token Limit"}


class TemperatureConfig(Config):
    """
    Controls the randomness of the model's output (0.0–1.0).
    Lower values produce more deterministic results.
    Higher values produce more varied responses.
    Cannot be used when Extended Thinking is enabled.
    """
    name: Literal["inputTemperature"] = "inputTemperature"
    value: float = 1.0
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Temperature"
        json_schema_extra = {"shortDescription": "Output Randomness"}


class MaxTokens(Config):
    """
    Maximum number of tokens in the model's response.
    Increase for longer outputs such as detailed captions or structured answers.
    Default is 3000.
    """
    name: Literal["maxTokens"] = "maxTokens"
    value: int = 3000
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Tokens"
        json_schema_extra = {"shortDescription": "Max Output Length"}


class MaxConcurrentRequests(Config):
    """
    Maximum number of API requests to run in parallel.
    Increase for higher throughput when processing multiple images.
    Default is 4.
    """
    name: Literal["maxConcurrentRequests"] = "maxConcurrentRequests"
    value: int = 4
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Concurrent Requests"
        json_schema_extra = {"shortDescription": "Parallel Requests"}


class TextPromptConfigs(Configs):
    inputPrompt: InputPrompt
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class TextPromptOutputs(Outputs):
    output: OutputText


class TextPromptRequest(Request):
    configs: TextPromptConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class TextPromptResponse(Response):
    outputs: TextPromptOutputs


class TextPrompt(Config):
    """
    Generates a text response based on a custom prompt.
    No image input is required for this task.
    Use for pure text tasks like content generation or Q&A.
    """
    name: Literal["TextPrompt"] = "TextPrompt"
    value: Union[TextPromptRequest, TextPromptResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text Prompt"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Text-only generation"}


class UnconstrainedConfigs(Configs):
    inputPrompt: InputPrompt
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class UnconstrainedInputs(Inputs):
    inputImage: InputImage


class UnconstrainedOutputs(Outputs):
    output: OutputText


class UnconstrainedRequest(Request):
    inputs: Optional[UnconstrainedInputs]
    configs: UnconstrainedConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class UnconstrainedResponse(Response):
    outputs: UnconstrainedOutputs


class Unconstrained(Config):
    """
    Analyzes an image using a fully custom prompt without predefined constraints.
    Provides maximum flexibility for open-ended visual analysis.
    Use when no structured output format is required.
    """
    name: Literal["Unconstrained"] = "Unconstrained"
    value: Union[UnconstrainedRequest, UnconstrainedResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Open Prompt"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Free-form image analysis"}

class OCRConfigs(Configs):
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class OCRInputs(Inputs):
    inputImage: InputImage


class OCROutputs(Outputs):
    output: OutputText


class OCRRequest(Request):
    inputs: Optional[OCRInputs]
    configs: OCRConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class OCRResponse(Response):
    outputs: OCROutputs


class OCR(Config):
    """
    Extracts all text present in an image using optical character recognition.
    Returns the detected text as a plain string.
    Works on documents, signs, labels, and handwritten content.
    """
    name: Literal["OCR"] = "OCR"
    value: Union[OCRRequest, OCRResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text Recognition (OCR)"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Text extraction from images"}

class VQAConfigs(Configs):
    inputPrompt: InputPrompt
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class VQAInputs(Inputs):
    inputImage: InputImage


class VQAOutputs(Outputs):
    output: OutputText


class VQARequest(Request):
    inputs: Optional[VQAInputs]
    configs: VQAConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class VQAResponse(Response):
    outputs: VQAOutputs


class VisualQuestionAnswering(Config):
    """
    Answers a specific question about the content of an image.
    Provide a question as the prompt and receive a targeted answer.
    Useful for structured visual inspection and querying.
    """
    name: Literal["VisualQuestionAnswering"] = "VisualQuestionAnswering"
    value: Union[VQARequest, VQAResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Visual Question Answering"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Image question answering"}



class CaptionConfigs(Configs):
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class CaptionInputs(Inputs):
    inputImage: InputImage


class CaptionOutputs(Outputs):
    output: OutputText


class CaptionRequest(Request):
    inputs: Optional[CaptionInputs]
    configs: CaptionConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class CaptionResponse(Response):
    outputs: CaptionOutputs


class ShortCaption(Configs):
    """
    Generates a concise one or two sentence caption describing an image.
    Suitable for labeling, quick summaries, or image metadata.
    Use Detailed Captioning for richer, longer descriptions.
    """
    name: Literal["ShortCaption"] = "ShortCaption"
    value: Union[CaptionRequest, CaptionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Captioning (Short)"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Brief image description"}

class DetailedCaptionConfigs(Configs):
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class DetailedCaptionInputs(Inputs):
    inputImage: InputImage


class DetailedCaptionOutputs(Outputs):
    output: OutputText


class DetailedCaptionRequest(Request):
    inputs: Optional[DetailedCaptionInputs]
    configs: DetailedCaptionConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class DetailedCaptionResponse(Response):
    outputs: DetailedCaptionOutputs


class DetailedCaption(Config):
    """
    Generates a comprehensive multi-sentence description of an image.
    Covers objects, attributes, spatial relationships, and scene context.
    Use for thorough image documentation or accessibility alt-text.
    """
    name: Literal["DetailedCaption"] = "DetailedCaption"
    value: Union[DetailedCaptionRequest, DetailedCaptionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Captioning (Detailed)"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Rich image description"}


class ClassificationConfigs(Configs):
    inputClasses: InputClasses
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class ClassificationInputs(Inputs):
    inputImage: InputImage


class ClassificationOutputs(Outputs):
    output: OutputText
    classes: Classes


class ClassificationRequest(Request):
    inputs: Optional[ClassificationInputs]
    configs: ClassificationConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ClassificationResponse(Response):
    outputs: ClassificationOutputs


class Classification(Config):
    """
    Assigns exactly one class label from a predefined list to an image.
    The model selects the single most appropriate class.
    Provide the list of valid classes using the Classes config.
    """
    name: Literal["Classification"] = "Classification"
    value: Union[ClassificationRequest, ClassificationResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Single-Label Classification"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Single class assignment"}

class MultiLabelConfigs(Configs):
    inputClasses: InputClasses
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class MultiLabelInputs(Inputs):
    inputImage: InputImage


class MultiLabelOutputs(Outputs):
    output: OutputText
    classes: Classes


class MultiLabelRequest(Request):
    inputs: Optional[MultiLabelInputs]
    configs: MultiLabelConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class MultiLabelResponse(Response):
    outputs: MultiLabelOutputs


class MultiLabel(Config):
    """
    Assigns one or more class labels from a predefined list to an image.
    The model selects all classes that apply to the image.
    Provide the list of valid classes using the Classes config.
    """
    name: Literal["MultiLabel"] = "MultiLabel"
    value: Union[MultiLabelRequest, MultiLabelResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Multi-Label Classification"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Multiple class assignment"}

class ObjectDetectionConfigs(Configs):
    inputClasses: InputClasses
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class ObjectDetectionInputs(Inputs):
    inputImage: InputImage


class ObjectDetectionOutputs(Outputs):
    output: OutputText
    classes: Classes


class ObjectDetectionRequest(Request):
    inputs: Optional[ObjectDetectionInputs]
    configs: ObjectDetectionConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ObjectDetectionResponse(Response):
    outputs: ObjectDetectionOutputs


class ObjectDetection(Config):
    """
    Detects and identifies objects in an image from a predefined class list.
    Returns detected class names found within the image.
    Provide the list of target classes using the Classes config.
    """
    name: Literal["ObjectDetection"] = "ObjectDetection"
    value: Union[ObjectDetectionRequest, ObjectDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Object Detection"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Detect objects by class"}

class StructuredAnsweringConfigs(Configs):
    inputPrompt: InputPrompt
    apiProvider: APIProvider
    inputModelVersion: InputModelVersion
    extendedThinking: ExtendedThinking
    thinkingBudgetTokens: ThinkingBudgetTokens
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class StructuredAnsweringInputs(Inputs):
    inputImage: InputImage


class StructuredAnsweringOutputs(Outputs):
    output: OutputText


class StructuredAnsweringRequest(Request):
    inputs: Optional[StructuredAnsweringInputs]
    configs: StructuredAnsweringConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class StructuredAnsweringResponse(Response):
    outputs: StructuredAnsweringOutputs


class StructuredAnswering(Config):
    """
    Generates a structured output (e.g. JSON) based on a custom prompt and image.
    Use the prompt to define the desired output schema or format.
    Ideal for extracting structured data from visual content.
    """
    name: Literal["StructuredAnswering"] = "StructuredAnswering"
    value: Union[StructuredAnsweringRequest, StructuredAnsweringResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Structured Output Generation"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Structured data extraction"}


class ConfigExecutor(Config):
    """
    Select the vision task to perform on the input image.
    Each task has its own prompt, output format, and configuration options.
    Choose the task that best matches your use case.
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[
        TextPrompt,
        Unconstrained,
        OCR,
        VisualQuestionAnswering,
        ShortCaption,
        DetailedCaption,
        Classification,
        MultiLabel,
        ObjectDetection,
        StructuredAnswering,
    ]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"shortDescription": "Select Vision Task"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["AnthropicClaude"] = "AnthropicClaude"
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
