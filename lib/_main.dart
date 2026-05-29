import 'package:flutter/material.dart';
import 'package:llamadart/llamadart.dart';
import 'package:flutter_background/flutter_background.dart';
import 'dart:async';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final androidConfig = FlutterBackgroundAndroidConfig(
    notificationTitle: "flutter_background example app",
    notificationText:
    "Background notification for keeping the example app running in the background",
    notificationImportance: AndroidNotificationImportance.normal,
    notificationIcon: AndroidResource(
      name: 'background_icon',
      defType: 'drawable',
    ), // Default is ic_launcher from folder mipmap
  );

  await FlutterBackground.initialize(androidConfig: androidConfig);
  await FlutterBackground.enableBackgroundExecution();

  LlamaEngine.configureLogging(level: LlamaLogLevel.debug);

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Local LLM with Llama',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Local LLM'),
    );
  }
}

enum AppState { idle, downloading, loading, processing, recording }

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class PerformanceMonitor extends StatefulWidget {
  const PerformanceMonitor({super.key});

  @override
  State<PerformanceMonitor> createState() => _PerformanceMonitorState();
}

class _PerformanceMonitorState extends State<PerformanceMonitor> {
  String _memoryRss = '0 MB';
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      _updateMetrics();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _updateMetrics() {
    final rss = ProcessInfo.currentRss;

    if (mounted) {
      setState(() {
        _memoryRss = '${(rss / 1024 / 1024).toStringAsFixed(1)} MB';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [_buildMetric('App Memory (RSS)', _memoryRss)],
      ),
    );
  }

  Widget _buildMetric(String label, String value) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$label: ',
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            color: Colors.greenAccent,
            fontSize: 12,
            fontFamily: 'monospace',
          ),
        ),
      ],
    );
  }
}

class ModelOption {
  final String name;
  final String url;

  const ModelOption(this.name, this.url);
}

const List<ModelOption> availableModels = [
  ModelOption(
    'Qwen2.5 0.5B Q4',
    'hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf',
  ),
  ModelOption(
    'Qwen3.5 0.8B Q4',
    'hf://unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf',
  ),
  ModelOption(
    'FunctionGemma Q4',
    'hf://unsloth/functiongemma-270m-it-GGUF/functiongemma-270m-it-Q4_K_M.gguf',
  ),
];

class _MyHomePageState extends State<MyHomePage> {
  final TextEditingController _controller = TextEditingController();
  AppState _state = AppState.idle;
  double _downloadProgress = 0;
  String _response = '';
  LlamaEngine? engine;
  ChatSession? session;
  ModelOption _selectedModel = availableModels.first;
  final Stopwatch _generationStopwatch = Stopwatch();
  Timer? _generationTimer;
  String _elapsedTime = '';

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _generationTimer?.cancel();
    super.dispose();
  }

  void _startGenerationTimer() {
    _generationStopwatch.reset();
    _generationStopwatch.start();
    _elapsedTime = '0.0s';
    _generationTimer?.cancel();
    _generationTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
      if (mounted) {
        setState(() {
          final ms = _generationStopwatch.elapsedMilliseconds;
          _elapsedTime = '${(ms / 1000).toStringAsFixed(1)}s';
        });
      }
    });
  }

  void _stopGenerationTimer() {
    _generationStopwatch.stop();
    _generationTimer?.cancel();
    _generationTimer = null;
    final ms = _generationStopwatch.elapsedMilliseconds;
    _elapsedTime = '${(ms / 1000).toStringAsFixed(1)}s';
  }

  Future<String> getPersistentCacheDir() async {
    final appDir =
    await getApplicationSupportDirectory(); // or getApplicationDocumentsDirectory()
    final modelCache = Directory('${appDir.path}/llamadart_models');
    await modelCache.create(recursive: true);
    return modelCache.path;
  }

  Future<void> _loadModel() async {
    try {
      setState(() {
        _state = AppState.loading;
        _downloadProgress = 0;
      });

      if (engine != null) {
        await engine!.dispose();
      }

      engine = LlamaEngine(LlamaBackend());
      session = ChatSession(engine!, maxContextTokens: 4096);

      final cacheDir = await getPersistentCacheDir();

      await engine!.loadModelSource(
        modelParams: ModelParams(contextSize: 4096),
        ModelSource.parse(_selectedModel.url),
        options: ModelLoadOptions(
          cachePolicy: ModelCachePolicy.preferCached,
          cacheDirectory: cacheDir,
        ),
        onProgress: (progress) {
          final fraction = progress.fraction;
          if (fraction != null) {
            setState(() {
              _downloadProgress = fraction * 100;
            });
          }
        },
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    } finally {
      setState(() {
        _state = AppState.idle;
      });
    }
  }

  Future<void> _cancelResponse() async {
    if (_state == AppState.processing) {
      engine?.cancelGeneration();
    }
    _stopGenerationTimer();
    setState(() {
      _state = AppState.idle;
    });
  }

  Future<void> _generateResponseFromText(String text) async {
    if (text.isEmpty) return;
    await _outputResponse(text);
  }

  Future<void> _outputResponse(String query) async {
    if (_state != AppState.idle || engine == null || session == null) return;

    try {
      if (mounted) {
        setState(() {
          _state = AppState.processing;
          _response = '';
        });
      }
      _startGenerationTimer();

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
            final location = params.getRequiredString('location');
            int temperature = 20;
            if (location == "Tunisia") temperature = 30;
            return {
              'location': location,
              'temperature': temperature,
              'unit': 'celsius',
            };
          },
        ),
        ToolDefinition(
          name: 'get_current_time',
          description: 'Get the current local time',
          parameters: [],
          handler: (params) async => {'time': DateTime.now().toIso8601String()},
        ),
      ];

      bool hasToolCalls = true;
      List<LlamaContentPart> parts = [LlamaTextContent(query)];

      while (hasToolCalls) {
        hasToolCalls = false;

        await for (final chunk in session!.create(
          parts,
          tools: tools,
          enableThinking: true,
          parallelToolCalls: true,
          toolChoice: ToolChoice.auto,
          params: GenerationParams(maxTokens: 1024, temp: 0.1),
        )) {
          if (_state != AppState.processing) break;

          final delta = chunk.choices.first.delta;

          // 1. Handle Thinking (Reasoning)
          if (delta.thinking != null && delta.thinking!.isNotEmpty) {
            if (mounted) setState(() => _response += delta.thinking!);
          }

          // 2. Handle Natural Text
          if (delta.content != null && delta.content!.isNotEmpty) {
            if (mounted) setState(() => _response += delta.content!);
          }

          // 3. Handle Tool Calls
          if (delta.toolCalls != null) {
            hasToolCalls = true;
            for (final tc in delta.toolCalls!) {
              if (tc.function?.name != null) {
                if (mounted) {
                  setState(
                        () => _response += '\n[Tool: ${tc.function!.name}]\n',
                  );
                }
              }
            }
          }
        }

        if (_state != AppState.processing) break;

        if (hasToolCalls) {
          parts = []; // Continue from history

          final lastMsg = session!.history.last;
          final toolCalls = lastMsg.parts.whereType<LlamaToolCallContent>();

          for (final tc in toolCalls) {
            final tool = tools.firstWhere(
                  (t) => t.name == tc.name,
              orElse: () => throw "Tool not found: ${tc.name}",
            );
            final result = await tool.invoke(tc.arguments);

            session!.addMessage(
              LlamaChatMessage.withContent(
                role: LlamaChatRole.tool,
                content: [
                  LlamaToolResultContent(
                    id: tc.id,
                    name: tc.name,
                    result: result,
                  ),
                ],
              ),
            );

            if (mounted) setState(() => _response += '[Result: $result]\n');
          }
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    } finally {
      _stopGenerationTimer();
      if (mounted) setState(() => _state = AppState.idle);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_sweep),
            onPressed: () => setState(() => session?.reset()),
            tooltip: 'Clear Chat',
          ),
          const Padding(
            padding: EdgeInsets.only(right: 8.0),
            child: PerformanceMonitor(),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            DropdownButton<ModelOption>(
              isExpanded: true,
              value: _selectedModel,
              items: availableModels.map((ModelOption model) {
                return DropdownMenuItem<ModelOption>(
                  value: model,
                  child: Text(model.name),
                );
              }).toList(),
              onChanged: (ModelOption? newValue) {
                if (newValue != null && newValue != _selectedModel) {
                  if (_state == AppState.processing) {
                    engine?.cancelGeneration();
                  }
                  setState(() {
                    _selectedModel = newValue;
                  });
                  _loadModel();
                }
              },
            ),
            const SizedBox(height: 16),
            if (_state == AppState.loading) ...[
              const SizedBox(height: 8),
              LinearProgressIndicator(value: _downloadProgress / 100),
              Text('Downloading: ${_downloadProgress.toStringAsFixed(1)}%'),
            ],
            const SizedBox(height: 16),
            TextField(
              controller: _controller,
              decoration: const InputDecoration(
                labelText: 'Enter prompt',
                border: OutlineInputBorder(),
              ),
              maxLines: null,
              enabled: _state == AppState.idle,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _state == AppState.idle
                        ? () => _generateResponseFromText(_controller.text)
                        : null,
                    child:
                    _state == AppState.loading ||
                        _state == AppState.processing
                        ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                        : const Text('Generate'),
                  ),
                ),
                const SizedBox(width: 8),
                if (_state == AppState.loading || _state == AppState.processing)
                  IconButton(
                    onPressed: _cancelResponse,
                    icon: const Icon(Icons.cancel, color: Colors.red),
                    tooltip: 'Cancel Response',
                  ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const Text(
                  'Response:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                if (_elapsedTime.isNotEmpty) ...[
                  const SizedBox(width: 8),
                  Text(
                    _elapsedTime,
                    style: TextStyle(
                      color: _state == AppState.processing
                          ? Colors.orange
                          : Colors.green,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'monospace',
                    ),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 8),
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: ListView.builder(
                  itemCount:
                  (session?.history.length ?? 0) +
                      (_state == AppState.processing ? 1 : 0),
                  itemBuilder: (context, index) {
                    if (index < (session?.history.length ?? 0)) {
                      final msg = session!.history[index];
                      if (msg.role == LlamaChatRole.system ||
                          msg.role == LlamaChatRole.tool) {
                        return const SizedBox.shrink();
                      }

                      final isUser = msg.role == LlamaChatRole.user;
                      return Align(
                        alignment: isUser
                            ? Alignment.centerRight
                            : Alignment.centerLeft,
                        child: Container(
                          margin: const EdgeInsets.symmetric(vertical: 4),
                          padding: const EdgeInsets.all(8),
                          constraints: BoxConstraints(
                            maxWidth: MediaQuery.of(context).size.width * 0.8,
                          ),
                          decoration: BoxDecoration(
                            color: isUser
                                ? Colors.blue.shade100
                                : Colors.grey.shade200,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(msg.content),
                        ),
                      );
                    } else {
                      return Align(
                        alignment: Alignment.centerLeft,
                        child: Container(
                          margin: const EdgeInsets.symmetric(vertical: 4),
                          padding: const EdgeInsets.all(8),
                          constraints: BoxConstraints(
                            maxWidth: MediaQuery.of(context).size.width * 0.8,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.grey.shade200,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            _response.isEmpty ? 'Thinking...' : _response,
                          ),
                        ),
                      );
                    }
                  },
                ),
              ),
            ),
            if (_state != AppState.idle)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Center(
                  child: Text(
                    'Status: ${_state.name.toUpperCase()}',
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.primary,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
