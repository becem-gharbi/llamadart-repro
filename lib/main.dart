import 'dart:io';
import 'package:flutter/material.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path_provider/path_provider.dart';

const LlamaLogLevel logLevel = LlamaLogLevel.error;

const modelUrl =
    //'hf://unsloth/functiongemma-270m-it-GGUF/functiongemma-270m-it-Q4_K_M.gguf';
    //'hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf';
    'hf://Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_Q4_k_m.gguf';

const prompts = [
  'What is the weather temperature in Tunis',
  'What is the weather humidity in Tunis',
  'What is the weather temperature and humidity in Tunis',
  'What is the weather temperature in Tunis and Bizerte',
  'What is the weather temperature in Tunis and What is the weather temperature in Bizerte',
];

final tools = [
  ToolDefinition(
    name: 'get_weather_temperature',
    description: 'Get current weather temperature for a location',
    parameters: [
      ToolParam.string(
        'location',
        description: 'Location name',
        required: true,
      ),
    ],
    handler: (params) async {
      final loc = params.getRequiredString('location');
      return {'location': loc, 'temperature': 25, 'unit': 'celsius'};
    },
  ),
  ToolDefinition(
    name: 'get_weather_humidity',
    description: 'Get current weather humidity for a location',
    parameters: [
      ToolParam.string(
        'location',
        description: 'Location name',
        required: true,
      ),
    ],
    handler: (params) async {
      final loc = params.getRequiredString('location');
      return {'location': loc, 'humidity': 60, 'unit': 'percent'};
    },
  ),
];

Future<void> generateResponse(ChatSession session, String prompt) async {
  debugPrint('\n[prompt] $prompt');

  bool hasToolCalls = true;
  List<LlamaContentPart> parts = [LlamaTextContent(prompt)];

  while (hasToolCalls) {
    hasToolCalls = false;

    final stream = session.create(
      parts,
      tools: tools,
      enableThinking: true,
      parallelToolCalls: true,
      toolChoice: ToolChoice.auto,
    );

    await for (final chunk in stream) {
      if (chunk.choices.first.delta.toolCalls != null) {
        hasToolCalls = true;
      }
    }

    if (hasToolCalls) {
      parts = [];
      final lastMsg = session.history.last;
      final toolCalls = lastMsg.parts.whereType<LlamaToolCallContent>();

      for (final tc in toolCalls) {
        final tool = tools.firstWhere((t) => t.name == tc.name);
        final result = await tool.invoke(tc.arguments);

        session.addMessage(
          LlamaChatMessage.withContent(
            role: LlamaChatRole.tool,
            content: [
              LlamaToolResultContent(id: tc.id, name: tc.name, result: result),
            ],
          ),
        );
      }
    }
  }

  debugPrint('[response] ${session.history.last.content}');
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  //****************************** Load model ******************************//
  final engine = LlamaEngine(LlamaBackend());
  await engine.setDartLogLevel(logLevel);
  await engine.setNativeLogLevel(logLevel);

  final appDir = await getApplicationSupportDirectory();
  final cacheDir = Directory('${appDir.path}/llamadart_models');
  await cacheDir.create(recursive: true);

  debugPrint('Loading model: $modelUrl...');
  await engine.loadModelSource(
    ModelSource.parse(modelUrl),
    options: ModelLoadOptions(
      cachePolicy: ModelCachePolicy.preferCached,
      cacheDirectory: cacheDir.path,
    ),
    onProgress: (progress) {
      if (progress.fraction != null) {
        debugPrint('Downloading ${progress.fraction! * 100}%');
      }
    },
  );
  debugPrint('\nModel loaded.');

  //****************************** start session ******************************//
  final session = ChatSession(engine);

  //****************************** generate responses ******************************//
  for (final prompt in prompts) {
    session.reset();
    await generateResponse(session, prompt);
  }

  //****************************** close ******************************//
  debugPrint('\nDone');
  session.reset();
  await engine.dispose();

  runApp(const MaterialApp());
}
