AWSTemplateFormatVersion: "2010-09-09"
Description: "AgentCore Lambda - Interaction Stream Processor (imports Redis + IAM stacks)"

Parameters:

  # ── Cross-stack references ──────────────────────────────────────────────────
  RedisStackName:
    Type: String
    Default: agentcore-redis
    Description: "Exact name of the deployed redis-stack (used for ImportValue)"

  IamStackName:
    Type: String
    Default: agentcore-iam
    Description: "Exact name of the deployed iam-stack (used for ImportValue)"

  # ── Lambda / networking ─────────────────────────────────────────────────────
  PrivateSubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Default: subnet-013d5e15202ffa634,subnet-0c5f6f1e835faeb9d

  RedisAuthToken:
    Type: String
    NoEcho: true
    Default: ""

  # ── Redis Layer (created from S3 zip) ───────────────────────────────────────
  RedisLayerS3Bucket:
    Type: String
    Default: kplrwl-s3-bucket-611184449569
    Description: "S3 bucket where the Redis layer zip is uploaded"

  RedisLayerS3Key:
    Type: String
    Default: LambdaLayers/redislayer.zip
    Description: "S3 key (path) to the Redis layer zip file"

  # ── Kinesis ─────────────────────────────────────────────────────────────────
  KinesisStreamArn:
    Type: String
    Default: arn:aws:kinesis:us-east-1:611184449569:stream/complaintsinteraction-stream

  # ── App config ───────────────────────────────────────────────────────────────
  AgentCoreEndpoint:
    Type: String
    Default: vpce-0da90a6d2df5b8927-e7jlclz4.bedrock-agentcore.us-east-1.vpce.amazonaws.com

  AgentCoreRuntimeArn:
    Type: String
    Default: arn:aws:bedrock-agentcore:us-east-1:611184449569:runtime/ail_new_flow_runtime-siEqhN3f4J

  ConfigS3Bucket:
    Type: String
    Default: kplrwl-s3-bucket-611184449569

  StateSchemaS3Key:
    Type: String
    Default: config/state_schema.json

  RulesS3Key:
    Type: String
    Default: config/orchestrator_rules.json

  UtteranceThreshold:
    Type: String
    Default: "5"

  DedupTtlSeconds:
    Type: String
    Default: "86400"

  SessionTtlSeconds:
    Type: String
    Default: "3600"

  OrchestratorTriggerInterval:
    Type: String
    Default: "4"

Conditions:
  UseAuthToken: !Not [!Equals [!Ref RedisAuthToken, ""]]

Resources:

  # ── Redis Lambda Layer (published from S3 zip) ──────────────────────────────
  RedisLambdaLayer:
    Type: AWS::Lambda::LayerVersion
    Properties:
      LayerName: "colleague-assist-orchestrator-redis-layer-dev"
      Description: "Redis client layer for Python 3.12"
      Content:
        S3Bucket: !Ref RedisLayerS3Bucket
        S3Key: !Ref RedisLayerS3Key
      CompatibleRuntimes:
        - python3.12

  # ── Lambda Function ─────────────────────────────────────────────────────────
  AgentRedisHealthCheckFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: "colleague-assist-orchestrator-interaction-stream-lambda-dev"
      Runtime: python3.12
      Handler: InteractionStream.lambda_handler
      Role: !ImportValue
        Fn::Sub: "${IamStackName}-LambdaExecutionRoleArn"
      Timeout: 60
      MemorySize: 256
      Layers:
        - !Ref RedisLambdaLayer        # ← references the layer created above
      VpcConfig:
        SubnetIds: !Ref PrivateSubnetIds
        SecurityGroupIds:
          - !ImportValue
            Fn::Sub: "${RedisStackName}-LambdaSecurityGroupId"
      Environment:
        Variables:
          REDIS_HOST: !ImportValue
            Fn::Sub: "${RedisStackName}-RedisPrimaryEndpoint"
          REDIS_PORT: !ImportValue
            Fn::Sub: "${RedisStackName}-RedisPort"
          REDIS_TLS: !ImportValue
            Fn::Sub: "${RedisStackName}-RedisTLS"
          REDIS_SSL: !ImportValue
            Fn::Sub: "${RedisStackName}-RedisTLS"
          REDIS_AUTH_TOKEN: !If [UseAuthToken, !Ref RedisAuthToken, ""]
          CONNECT_TIMEOUT: "2"
          SOCKET_TIMEOUT: "2"
          ORCHESTRATOR_TRIGGER_INTERVAL: !Ref OrchestratorTriggerInterval
          AGENTCORE_ENDPOINT: !Ref AgentCoreEndpoint
          AGENTCORE_RUNTIME_ARN: !Ref AgentCoreRuntimeArn
          CONFIG_S3_BUCKET: !Ref ConfigS3Bucket
          STATE_SCHEMA_S3_KEY: !Ref StateSchemaS3Key
          RULES_S3_KEY: !Ref RulesS3Key
          UTTERANCE_THRESHOLD: !Ref UtteranceThreshold
          DEDUP_TTL_SECONDS: !Ref DedupTtlSeconds
          SESSION_TTL_SECONDS: !Ref SessionTtlSeconds
      Code:
        S3Bucket: kplrwl-s3-bucket-611184449569
        S3Key: LambdaFunctions/InteractionStream.zip

  # ── Kinesis Event Source Mapping ────────────────────────────────────────────
  AgentKinesisEventMapping:
    Type: AWS::Lambda::EventSourceMapping
    Properties:
      EventSourceArn: !Ref KinesisStreamArn
      FunctionName: !Ref AgentRedisHealthCheckFunction
      Enabled: true
      StartingPosition: LATEST
      BatchSize: 100
      MaximumBatchingWindowInSeconds: 5
      BisectBatchOnFunctionError: true
      MaximumRetryAttempts: 2
      FunctionResponseTypes:
        - ReportBatchItemFailures

Outputs:

  RedisLayerArn:
    Description: "ARN of the Redis Lambda layer version"
    Value: !Ref RedisLambdaLayer
    Export:
      Name: !Sub "${AWS::StackName}-RedisLayerArn"

  LambdaFunctionName:
    Value: !Ref AgentRedisHealthCheckFunction
    Export:
      Name: !Sub "${AWS::StackName}-LambdaFunctionName"

  LambdaFunctionArn:
    Value: !GetAtt AgentRedisHealthCheckFunction.Arn
    Export:
      Name: !Sub "${AWS::StackName}-LambdaFunctionArn"
