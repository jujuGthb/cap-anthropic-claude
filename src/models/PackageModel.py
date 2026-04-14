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
    name: Literal["inputPrompt"] = "inputPrompt"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Prompt"
        json_schema_extra = {"shortDescription": "User Prompt"}


class InputApiKey(Config):
    """
    Enter your Anthropic API key.
    Keys starting with 'sk-ant-' are supported.
    You can also use the NovaVision proxy by selecting NovaVision.
    """
    name: Literal["inputApiKey"] = "inputApiKey"
    value: str
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "API Key"
        json_schema_extra = {"shortDescription": "Anthropic API Key"}


class AnthropicAPIConfigs(Configs):
    inputApiKey: InputApiKey


class AnthropicAPIOption(Config):
    name: Literal["Anthropic"] = "Anthropic"
    value: AnthropicAPIConfigs
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Anthropic API"
        json_schema_extra = {"target": "value"}


class NovaVisionAPIConfigs(Configs):
    pass


class NovaVisionOption(Config):
    name: Literal["NovaVision"] = "NovaVision"
    value: NovaVisionAPIConfigs
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "NovaVision"
        json_schema_extra = {"target": "value"}


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
    name: Literal["claude-opus-4-6"] = "claude-opus-4-6"
    value: Literal["claude-opus-4-6"] = "claude-opus-4-6"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Opus 4.6"


class VersionSonnet46(Config):
    name: Literal["claude-sonnet-4-6"] = "claude-sonnet-4-6"
    value: Literal["claude-sonnet-4-6"] = "claude-sonnet-4-6"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Sonnet 4.6"


class VersionSonnet45(Config):
    name: Literal["claude-sonnet-4-5"] = "claude-sonnet-4-5"
    value: Literal["claude-sonnet-4-5"] = "claude-sonnet-4-5"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Sonnet 4.5"


class VersionHaiku45(Config):
    name: Literal["claude-haiku-4-5"] = "claude-haiku-4-5"
    value: Literal["claude-haiku-4-5"] = "claude-haiku-4-5"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Haiku 4.5"


class VersionOpus45(Config):
    name: Literal["claude-opus-4-5"] = "claude-opus-4-5"
    value: Literal["claude-opus-4-5"] = "claude-opus-4-5"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Opus 4.5"


class VersionSonnet4(Config):
    name: Literal["claude-sonnet-4"] = "claude-sonnet-4"
    value: Literal["claude-sonnet-4"] = "claude-sonnet-4"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Sonnet 4"


class VersionOpus41(Config):
    name: Literal["claude-opus-4-1"] = "claude-opus-4-1"
    value: Literal["claude-opus-4-1"] = "claude-opus-4-1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Opus 4.1"


class VersionOpus4(Config):
    name: Literal["claude-opus-4"] = "claude-opus-4"
    value: Literal["claude-opus-4"] = "claude-opus-4"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Claude Opus 4"


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
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ExtendedThinkingFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


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


class TextPromptExecutor(Config):
    name: Literal["TextPromptExecutor"] = "TextPromptExecutor"
    value: Union[TextPromptRequest, TextPromptResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text Prompt"
        json_schema_extra = {"target": {"value": 0}}


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


class UnconstrainedExecutor(Config):
    name: Literal["UnconstrainedExecutor"] = "UnconstrainedExecutor"
    value: Union[UnconstrainedRequest, UnconstrainedResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Open Prompt"
        json_schema_extra = {"target": {"value": 0}}



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


class OCRExecutor(Config):
    name: Literal["OCRExecutor"] = "OCRExecutor"
    value: Union[OCRRequest, OCRResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text Recognition (OCR)"
        json_schema_extra = {"target": {"value": 0}}



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


class VQAExecutor(Config):
    name: Literal["VQAExecutor"] = "VQAExecutor"
    value: Union[VQARequest, VQAResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Visual Question Answering"
        json_schema_extra = {"target": {"value": 0}}



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


class CaptionExecutor(Config):
    name: Literal["CaptionExecutor"] = "CaptionExecutor"
    value: Union[CaptionRequest, CaptionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Captioning (Short)"
        json_schema_extra = {"target": {"value": 0}}


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


class DetailedCaptionExecutor(Config):
    name: Literal["DetailedCaptionExecutor"] = "DetailedCaptionExecutor"
    value: Union[DetailedCaptionRequest, DetailedCaptionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Captioning (Detailed)"
        json_schema_extra = {"target": {"value": 0}}



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


class ClassificationExecutor(Config):
    name: Literal["ClassificationExecutor"] = "ClassificationExecutor"
    value: Union[ClassificationRequest, ClassificationResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Single-Label Classification"
        json_schema_extra = {"target": {"value": 0}}




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


class MultiLabelExecutor(Config):
    name: Literal["MultiLabelExecutor"] = "MultiLabelExecutor"
    value: Union[MultiLabelRequest, MultiLabelResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Multi-Label Classification"
        json_schema_extra = {"target": {"value": 0}}


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


class ObjectDetectionExecutor(Config):
    name: Literal["ObjectDetectionExecutor"] = "ObjectDetectionExecutor"
    value: Union[ObjectDetectionRequest, ObjectDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Object Detection"
        json_schema_extra = {"target": {"value": 0}}



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


class StructuredAnsweringExecutor(Config):
    name: Literal["StructuredAnsweringExecutor"] = "StructuredAnsweringExecutor"
    value: Union[StructuredAnsweringRequest, StructuredAnsweringResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Structured Output Generation"
        json_schema_extra = {"target": {"value": 0}}



class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[
        TextPromptExecutor,
        UnconstrainedExecutor,
        OCRExecutor,
        VQAExecutor,
        CaptionExecutor,
        DetailedCaptionExecutor,
        ClassificationExecutor,
        MultiLabelExecutor,
        ObjectDetectionExecutor,
        StructuredAnsweringExecutor,
    ]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["Claude"] = "Claude"
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"