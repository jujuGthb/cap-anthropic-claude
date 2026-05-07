from sdks.novavision.src.helper.package import PackageHelper
from capsules.AnthropicClaude.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    OutputText,
    Classes,
    TextPrompt,
    TextPromptResponse,
    TextPromptOutputs,
    Unconstrained,
    UnconstrainedResponse,
    UnconstrainedOutputs,
    OCR,
    OCRResponse,
    OCROutputs,
    VisualQuestionAnswering,
    VQAResponse,
    VQAOutputs,
    ShortCaption,
    CaptionResponse,
    CaptionOutputs,
    DetailedCaption,
    DetailedCaptionResponse,
    DetailedCaptionOutputs,
    Classification,
    ClassificationResponse,
    ClassificationOutputs,
    MultiLabel,
    MultiLabelResponse,
    MultiLabelOutputs,
    ObjectDetection,
    ObjectDetectionResponse,
    ObjectDetectionOutputs,
    StructuredAnswering,
    StructuredAnsweringResponse,
    StructuredAnsweringOutputs,
)


def build_response_text_prompt(context):
    output = OutputText(value=context.claude_text)
    outputs = TextPromptOutputs(output=output)
    response = TextPromptResponse(outputs=outputs)
    executor = TextPrompt(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_unconstrained(context):
    output = OutputText(value=context.claude_text)
    outputs = UnconstrainedOutputs(output=output)
    response = UnconstrainedResponse(outputs=outputs)
    executor = Unconstrained(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_ocr(context):
    output = OutputText(value=context.claude_text)
    outputs = OCROutputs(output=output)
    response = OCRResponse(outputs=outputs)
    executor = OCR(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_vqa(context):
    output = OutputText(value=context.claude_text)
    outputs = VQAOutputs(output=output)
    response = VQAResponse(outputs=outputs)
    executor = VisualQuestionAnswering(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_caption(context):
    output = OutputText(value=context.claude_text)
    outputs = CaptionOutputs(output=output)
    response = CaptionResponse(outputs=outputs)
    executor = ShortCaption(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_detailed_caption(context):
    output = OutputText(value=context.claude_text)
    outputs = DetailedCaptionOutputs(output=output)
    response = DetailedCaptionResponse(outputs=outputs)
    executor = DetailedCaption(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_classification(context):
    output = OutputText(value=context.claude_text)
    classes = Classes(value=context.claude_classes if context.claude_classes else [])
    outputs = ClassificationOutputs(output=output, classes=classes)
    response = ClassificationResponse(outputs=outputs)
    executor = Classification(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_multi_label(context):
    output = OutputText(value=context.claude_text)
    classes = Classes(value=context.claude_classes if context.claude_classes else [])
    outputs = MultiLabelOutputs(output=output, classes=classes)
    response = MultiLabelResponse(outputs=outputs)
    executor = MultiLabel(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_object_detection(context):
    output = OutputText(value=context.claude_text)
    classes = Classes(value=context.claude_classes if context.claude_classes else [])
    outputs = ObjectDetectionOutputs(output=output, classes=classes)
    response = ObjectDetectionResponse(outputs=outputs)
    executor = ObjectDetection(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)


def build_response_structured_answering(context):
    output = OutputText(value=context.claude_text)
    outputs = StructuredAnsweringOutputs(output=output)
    response = StructuredAnsweringResponse(outputs=outputs)
    executor = StructuredAnswering(value=response)
    configExecutor = ConfigExecutor(value=executor)
    packageConfigs = PackageConfigs(executor=configExecutor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)

    return package.build_model(context)