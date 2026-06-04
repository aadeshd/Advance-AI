redis
AWSTemplateFormatVersion: "2010-09-09"
Description: "AgentCore VPC - Redis (ElastiCache + Security Groups + Subnet Group)"

Parameters:

  VpcId:
    Type: AWS::EC2::VPC::Id
    Default: vpc-0f35f404b89ae206c

  PrivateSubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Default: subnet-013d5e15202ffa634,subnet-0c5f6f1e835faeb9d

  RedisPort:
    Type: Number
    Default: 6379

  RedisNodeType:
    Type: String
    Default: cache.t3.small

  NumCacheClusters:
    Type: Number
    Default: 1

  EnableTransitEncryption:
    Type: String
    AllowedValues: ["true", "false"]
    Default: "false"

  RedisAuthToken:
    Type: String
    NoEcho: true
    Default: ""

  # The Lambda SG from the lambda stack needs ingress into Redis.
  # Pass it here if the lambda stack is deployed first, or use a separate
  # ingress rule update after both stacks are up.
  # Keeping the existing hardcoded SG reference for parity with original template.
  ExternalSourceSecurityGroupId:
    Type: String
    Default: sg-07914e60aee7a0c4f
    Description: "Additional SG allowed to reach Redis (e.g. bastion / existing agent SG)"

Conditions:
  UseTLS: !Equals [!Ref EnableTransitEncryption, "true"]
  UseAuthToken: !Not [!Equals [!Ref RedisAuthToken, ""]]

Resources:

  AgentLambdaSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: "Agent Lambda SG"
      VpcId: !Ref VpcId
      SecurityGroupEgress:
        - IpProtocol: "-1"
          CidrIp: 0.0.0.0/0

  AgentRedisSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: "Agent Redis SG"
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: !Ref RedisPort
          ToPort: !Ref RedisPort
          SourceSecurityGroupId: !Ref AgentLambdaSecurityGroup
        - IpProtocol: tcp
          FromPort: !Ref RedisPort
          ToPort: !Ref RedisPort
          SourceSecurityGroupId: !Ref ExternalSourceSecurityGroupId

  AgentRedisSubnetGroup:
    Type: AWS::ElastiCache::SubnetGroup
    Properties:
      Description: "Agent Redis subnet group"
      SubnetIds: !Ref PrivateSubnetIds

  AgentRedisReplicationGroup:
    Type: AWS::ElastiCache::ReplicationGroup
    Properties:
      ReplicationGroupId: !Sub "agent-redis-${AWS::StackName}"
      ReplicationGroupDescription: "Agent Redis cluster"
      Engine: redis
      CacheNodeType: !Ref RedisNodeType
      NumCacheClusters: !Ref NumCacheClusters
      AutomaticFailoverEnabled: false
      Port: !Ref RedisPort
      CacheSubnetGroupName: !Ref AgentRedisSubnetGroup
      SecurityGroupIds:
        - !Ref AgentRedisSecurityGroup
      TransitEncryptionEnabled: !If [UseTLS, true, false]
      AuthToken: !If [UseAuthToken, !Ref RedisAuthToken, !Ref "AWS::NoValue"]

Outputs:

  RedisPrimaryEndpoint:
    Description: "Redis primary endpoint address"
    Value: !GetAtt AgentRedisReplicationGroup.PrimaryEndPoint.Address
    Export:
      Name: !Sub "${AWS::StackName}-RedisPrimaryEndpoint"

  RedisPort:
    Description: "Redis port"
    Value: !Ref RedisPort
    Export:
      Name: !Sub "${AWS::StackName}-RedisPort"

  RedisTLS:
    Description: "Whether TLS is enabled"
    Value: !If [UseTLS, "true", "false"]
    Export:
      Name: !Sub "${AWS::StackName}-RedisTLS"

  AgentLambdaSecurityGroupId:
    Description: "Security group ID to attach to the Lambda function"
    Value: !Ref AgentLambdaSecurityGroup
    Export:
      Name: !Sub "${AWS::StackName}-LambdaSecurityGroupId"
iam
AWSTemplateFormatVersion: "2010-09-09"
Description: "AgentCore IAM - Lambda Execution Role"

Parameters:

  KinesisStreamArn:
    Type: String
    Default: arn:aws:kinesis:us-east-1:611184449569:stream/complaintsinteraction-stream

  AgentRuntimeId:
    Type: String
    Default: ail_orchestrator_runtime_stable-JhF5vuHmgW

  AILOrchestratorInvokeLogGroup:
    Type: String
    Default: /aws/bedrock-agentcore/runtimes/ail_orchestrator_runtime_stable-JhF5vuHmgW-DEFAULT

Resources:

  AgentLambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: "colleague-assist-orchestrator-interaction-stream-iam-dev"
      PermissionsBoundary: arn:aws:iam::611184449569:policy/core-ServiceRolePermissionsBoundary
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        - arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole

      Policies:
        - PolicyName: KinesisReadAccess
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - kinesis:GetRecords
                  - kinesis:GetShardIterator
                  - kinesis:DescribeStream
                  - kinesis:DescribeStreamSummary
                  - kinesis:ListShards
                  - kinesis:ListStreams
                Resource: !Ref KinesisStreamArn
              - Effect: Allow
                Action:
                  - kinesis:ListStreams
                Resource: "*"

        - PolicyName: InvokeAgentCoreRuntime
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Sid: WriteLogs
                Effect: Allow
                Action:
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource:
                  - !Sub "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:${AILOrchestratorInvokeLogGroup}"
                  - !Sub "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:${AILOrchestratorInvokeLogGroup}:*"

              - Sid: VpcEni
                Effect: Allow
                Action:
                  - ec2:CreateNetworkInterface
                  - ec2:DescribeNetworkInterfaces
                  - ec2:DeleteNetworkInterface
                  - ec2:DescribeSubnets
                  - ec2:DescribeSecurityGroups
                  - ec2:DescribeVpcs
                Resource: "*"

              - Sid: InvokeAgentRuntime
                Effect: Allow
                Action:
                  - bedrock-agentcore:InvokeAgentRuntime
                Resource:
                  - !Sub "arn:aws:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:runtime/${AgentRuntimeId}"
                  - !Sub "arn:aws:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:runtime/${AgentRuntimeId}/runtime-endpoint/DEFAULT"

              - Sid: StopAgentCoreSession
                Effect: Allow
                Action:
                  - bedrock-agentcore:StopRuntimeSession
                Resource:
                  - !Sub "arn:aws:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:runtime/${AgentRuntimeId}"
                  - !Sub "arn:aws:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:runtime/${AgentRuntimeId}/runtime-endpoint/DEFAULT"

        - PolicyName: BedrockInvokePolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Sid: BedrockInvoke
                Effect: Allow
                Action:
                  - bedrock:InvokeModel
                  - bedrock:InvokeModelWithResponseStream
                  - bedrock:Converse
                  - bedrock:ConverseStream
                Resource: "*"

        - PolicyName: S3ReadPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Sid: S3ReadAccess
                Effect: Allow
                Action:
                  - s3:GetObject
                Resource: "*"

        - PolicyName: SSMReadPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Sid: SSMParameterRead
                Effect: Allow
                Action:
                  - ssm:GetParameter
                  - ssm:GetParameters
                  - ssm:GetParametersByPath
                Resource:
                  - !Sub "arn:aws:ssm:${AWS::Region}:${AWS::AccountId}:parameter/agentcore/*"

Outputs:

  LambdaExecutionRoleArn:
    Description: "ARN of the Lambda execution role"
    Value: !GetAtt AgentLambdaExecutionRole.Arn
    Export:
      Name: !Sub "${AWS::StackName}-LambdaExecutionRoleArn"

  LambdaExecutionRoleName:
    Description: "Name of the Lambda execution role"
    Value: !Ref AgentLambdaExecutionRole
    Export:
      Name: !Sub "${AWS::StackName}-LambdaExecutionRoleName"


lambda

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

  RedisLayerArn:
    Type: String
    Default: arn:aws:lambda:us-east-1:611184449569:layer:redislayer:1

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
        - !Ref RedisLayerArn
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

  LambdaFunctionName:
    Value: !Ref AgentRedisHealthCheckFunction
    Export:
      Name: !Sub "${AWS::StackName}-LambdaFunctionName"

  LambdaFunctionArn:
    Value: !GetAtt AgentRedisHealthCheckFunction.Arn
    Export:
      Name: !Sub "${AWS::StackName}-LambdaFunctionArn"

