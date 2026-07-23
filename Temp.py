AWSTemplateFormatVersion: "2010-09-09"
Description: >
  Colleague Assist Orchestrator - provisions Redis, AgentCore Runtime and
  Interaction Stream Lambda. This parent template creates the required security
  groups, stores their IDs in SSM, creates the AgentCore execution role, and
  creates a normal Lambda function with Kinesis trigger inside the same VPC.
  Existing subnet SSM parameter paths are supplied as parameters; this template
  does not create subnets.

Parameters:
  AppName:
    Type: String
    Default: col-ast
    Description: Application name used as naming prefix.

  Environment:
    Type: String
    Default: sit
    AllowedValues:
      - dev
      - test
      - uat
      - staging
      - prod
      - sit
    Description: Deployment environment.

  PermissionsBoundaryArn:
    Type: String
    Default: arn:aws:iam::739275437751:policy/core-ServiceRolePermissionsBoundary
    Description: Permissions boundary required by organisation governance.

  # ---------------------------------------------------------------------------
  # Network inputs - existing SSM parameter paths; no subnets are created here.
  # ---------------------------------------------------------------------------
  VpcIdPath:
    Type: String
    Default: /app/network/VPCId
    Description: SSM parameter path for VPC ID, passed to Service Catalog products.

  VpcIdValue:
    Type: AWS::SSM::Parameter::Value<AWS::EC2::VPC::Id>
    Default: /app/network/VPCId
    Description: Resolved VPC ID used only by this parent template to create security groups.

  VpcCidrSsmPath:
    Type: String
    Default: /app/network/VPCCidr
    Description: SSM parameter path for VPC CIDR, passed to AgentCore product.

  ElastiCacheSubnet1:
    Type: String
    Default: /ail/network/vpce/subnet1
    Description: Existing SSM parameter path for Redis subnet 1.

  ElastiCacheSubnet2:
    Type: String
    Default: /ail/network/vpce/subnet2
    Description: Existing SSM parameter path for Redis subnet 2.

  LambdaSubnet1:
    Type: AWS::SSM::Parameter::Value<AWS::EC2::Subnet::Id>
    Default: /ail/network/vpce/subnet1
    Description: Existing SSM parameter path for Lambda subnet 1.

  LambdaSubnet2:
    Type: AWS::SSM::Parameter::Value<AWS::EC2::Subnet::Id>
    Default: /ail/network/vpce/subnet2
    Description: Existing SSM parameter path for Lambda subnet 2.

  AgentCoreSubnet1:
    Type: String
    Default: /ail/network/vpce/subnet1
    Description: Existing SSM parameter path for AgentCore subnet 1.

  AgentCoreSubnet2:
    Type: String
    Default: /ail/network/vpce/subnet2
    Description: Existing SSM parameter path for AgentCore subnet 2.

  RedisAdditionalSecurityGroupId:
    Type: String
    Default: /ail/network/vpce/sg-redis-col-ast
    Description: SSM parameter name where this template stores the Redis additional security group ID.

  LambdaSecurityGroupId:
    Type: String
    Default: /ail/network/vpce/sg-lambda-col-ast
    Description: SSM parameter name where this template stores the Lambda security group ID.

  AgentCoreAdditionalSecurityGroupId:
    Type: String
    Default: /ail/network/vpce/sg-runtime-col-ast
    Description: SSM parameter name where this template stores the AgentCore additional security group ID.

  # ---------------------------------------------------------------------------
  # Service Catalog product and provisioning artifact IDs
  # ---------------------------------------------------------------------------
  RedisProductId:
    Type: String
    Default: prod-hh6cktu2tec7u
    Description: Service Catalog product ID for the ElastiCache Redis product.

  RedisProvisioningArtifactId:
    Type: String
    Default: pa-j3isrx4urbww4
    Description: Service Catalog provisioning artifact ID for the Redis product.

  AgentCoreProductId:
    Type: String
    Default: prod-vuen5s5ulfrpe
    Description: Service Catalog product ID for the Bedrock AgentCore Runtime product.

  AgentCoreProvisioningArtifactId:
    Type: String
    Default: pa-it5epphcljtbe
    Description: Service Catalog provisioning artifact ID for the AgentCore product.

  # ---------------------------------------------------------------------------
  # Redis config
  # ---------------------------------------------------------------------------
  RedisPort:
    Type: Number
    Default: 6379

  RedisCacheNodeType:
    Type: String
    Default: cache.t3.small

  RedisEngineVersion:
    Type: String
    Default: "7.0"

  RedisNumNodeGroups:
    Type: Number
    Default: 1

  RedisReplicasPerNodeGroup:
    Type: Number
    Default: 0

  RedisMultiAZEnabled:
    Type: String
    AllowedValues:
      - "true"
      - "false"
    Default: "false"

  RedisSnapshotRetentionLimit:
    Type: Number
    Default: 5

  RedisLogFormat:
    Type: String
    Default: json

  RedisLogType:
    Type: String
    Default: slow-log

  RedisLogRetentionDays:
    Type: Number
    Default: 90

  RedisCacheLogsEnabled:
    Type: String
    AllowedValues:
      - "true"
      - "false"
    Default: "false"

  RedisTlsEnabled:
    Type: String
    AllowedValues:
      - "true"
      - "false"
    # FIX 2: cluster actually has transit encryption enabled; this must match
    # reality so REDIS_TLS passed to Lambda is correct and the client uses ssl=True.
    Default: "true"

  # ---------------------------------------------------------------------------
  # Lambda config
  # ---------------------------------------------------------------------------
  LambdaS3BucketName:
    Type: String
    Default: ail-orchestrator-artifacts-test-739275437751
    Description: S3 bucket holding Lambda code and config.

  LambdaDescription:
    Type: String
    Default: Colleague assist interaction stream lambda

  LambdaHandler:
    Type: String
    Default: InteractionStream.lambda_handler

  LambdaRuntime:
    Type: String
    Default: python3.12

  LambdaTimeout:
    Type: Number
    Default: 30

  LambdaArchitecture:
    Type: String
    Default: x86_64

  LambdaMemorySize:
    Type: Number
    Default: 128

  LambdaTracing:
    Type: String
    Default: Active

  LambdaCodeKeyName:
    Type: String
    Default: lambda/interaction-stream/lego/code/InteractionStreamCode.zip

  ConnectTimeoutSeconds:
    Type: Number
    Default: 2

  SocketTimeoutSeconds:
    Type: Number
    Default: 2

  DedupTtlSeconds:
    Type: Number
    Default: 86400

  OrchestratorTriggerIntervalSeconds:
    Type: Number
    Default: 10

  SessionTtlSeconds:
    Type: Number
    Default: 3600

  UtteranceThreshold:
    Type: Number
    Default: 5

  EventConfigS3Key:
    Type: String
    Default: lambda/interaction-stream/config/event_config.json

  RulesS3Key:
    Type: String
    Default: lambda/interaction-stream/config/orchestrator_rules.json

  StateSchemaS3Key:
    Type: String
    Default: lambda/interaction-stream/config/state_schema.json

  # ---------------------------------------------------------------------------
  # Kinesis trigger
  # ---------------------------------------------------------------------------
  KinesisStreamArn:
    Type: String
    Description: ARN of the Kinesis stream that triggers the Lambda.

  KinesisBatchSize:
    Type: Number
    Default: 100

  KinesisStartingPosition:
    Type: String
    AllowedValues:
      - LATEST
      - TRIM_HORIZON
      - AT_TIMESTAMP
    Default: LATEST

  # ---------------------------------------------------------------------------
  # AgentCore config
  # ---------------------------------------------------------------------------
  AgentCoreExecutionRoleName:
    Type: String
    Default: svc-agentcore-runtime-execution-role
    Description: IAM role name for Bedrock AgentCore Runtime execution.

  AgentCoreCodeS3Key:
    Type: String
    Default: orchestrator-out/package.zip

  AgentCoreCodeRuntime:
    Type: String
    Default: PYTHON_3_12

  AgentCoreCodeEntrypoint:
    Type: String
    Default: main.py

  AgentCoreProtocol:
    Type: String
    Default: HTTP

  AgentCoreBusinessUnit:
    Type: String
    Default: BUK

  AgentCoreConfigBasePath:
    Type: String
    Default: configs

  AgentCoreConfigCacheTtl:
    Type: Number
    Default: 3600

  AgentCoreLogLevel:
    Type: String
    Default: INFO


Resources:
  # ---------------------------------------------------------------------------
  # Security groups created in this parent stack and stored in SSM
  # ---------------------------------------------------------------------------
  RedisAdditionalSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: !Sub "${AppName}-${Environment}-redis-additional-sg"
      VpcId: !Ref VpcIdValue
      SecurityGroupEgress:
        - IpProtocol: "-1"
          CidrIp: 0.0.0.0/0
      Tags:
        - Key: Name
          Value: !Sub "${AppName}-${Environment}-redis-additional-sg"
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  LambdaSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: !Sub "${AppName}-${Environment}-lambda-sg"
      VpcId: !Ref VpcIdValue
      SecurityGroupEgress:
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
          Description: HTTPS for AWS services and bedrock-agentcore VPC endpoint
        - IpProtocol: tcp
          FromPort: 6379
          ToPort: 6379
          DestinationSecurityGroupId: !Ref RedisAdditionalSecurityGroup
          Description: Redis access
      Tags:
        - Key: Name
          Value: !Sub "${AppName}-${Environment}-lambda-sg"
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  AgentCoreAdditionalSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: !Sub "${AppName}-${Environment}-agentcore-additional-sg"
      VpcId: !Ref VpcIdValue
      SecurityGroupEgress:
        - IpProtocol: tcp
          FromPort: 6379
          ToPort: 6379
          DestinationSecurityGroupId: !Ref RedisAdditionalSecurityGroup
          Description: Allow AgentCore to access Redis
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
          Description: HTTPS for AWS services
      Tags:
        - Key: Name
          Value: !Sub "${AppName}-${Environment}-agentcore-additional-sg"
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  # ---------------------------------------------------------------------------
  # FIX 1: VPC Interface Endpoint (PrivateLink) for bedrock-agentcore.
  # Lambda's InvokeAgentRuntime call is a control-plane API call that never
  # touches the AgentCore Runtime's own VPC ENI/SG - it needs its own network
  # path out of Lambda's private subnets to the bedrock-agentcore service.
  # With no NAT Gateway in this VPC, PrivateLink is required.
  # ---------------------------------------------------------------------------
  BedrockAgentCoreEndpointSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: !Sub "${AppName}-${Environment}-agentcore-vpce-sg"
      VpcId: !Ref VpcIdValue
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          SourceSecurityGroupId: !Ref LambdaSecurityGroup
          Description: Allow Lambda to reach bedrock-agentcore PrivateLink endpoint
      Tags:
        - Key: Name
          Value: !Sub "${AppName}-${Environment}-agentcore-vpce-sg"
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  BedrockAgentCoreVpcEndpoint:
    Type: AWS::EC2::VPCEndpoint
    Properties:
      VpcId: !Ref VpcIdValue
      ServiceName: !Sub "com.amazonaws.${AWS::Region}.bedrock-agentcore"
      VpcEndpointType: Interface
      PrivateDnsEnabled: true
      SubnetIds:
        - !Ref LambdaSubnet1
        - !Ref LambdaSubnet2
      SecurityGroupIds:
        - !Ref BedrockAgentCoreEndpointSecurityGroup

  RedisAdditionalSecurityGroupParameter:
    Type: AWS::SSM::Parameter
    Properties:
      Name: !Ref RedisAdditionalSecurityGroupId
      Type: String
      Value: !Ref RedisAdditionalSecurityGroup
      Description: Redis additional security group ID.

  LambdaSecurityGroupParameter:
    Type: AWS::SSM::Parameter
    Properties:
      Name: !Ref LambdaSecurityGroupId
      Type: String
      Value: !Ref LambdaSecurityGroup
      Description: Lambda security group ID.

  AgentCoreAdditionalSecurityGroupParameter:
    Type: AWS::SSM::Parameter
    Properties:
      Name: !Ref AgentCoreAdditionalSecurityGroupId
      Type: String
      Value: !Ref AgentCoreAdditionalSecurityGroup
      Description: AgentCore additional security group ID.

  # ---------------------------------------------------------------------------
  # AgentCore Runtime execution role
  # ---------------------------------------------------------------------------
  AgentCoreRuntimeExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub "${AgentCoreExecutionRoleName}-${Environment}"
      PermissionsBoundary: !Ref PermissionsBoundaryArn
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Sid: AgentCoreAssumeRole
            Effect: Allow
            Principal:
              Service: bedrock-agentcore.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: AgentCoreRuntimeExecutionPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Sid: WriteLogs
                Effect: Allow
                Action:
                  - logs:CreateLogGroup
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource:
                  - !Sub "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:*"
                  - !GetAtt AgentCoreLogGroup.Arn
                  - !Sub "${AgentCoreLogGroup.Arn}:*"
              - Sid: InvokeAgentRuntime
                Effect: Allow
                Action:
                  - bedrock-agentcore:InvokeAgentRuntime
                Resource:
                  - !Sub "arn:aws:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:runtime/col_ast_orchestrator*"
              - Sid: StopAgentCoreSession
                Effect: Allow
                Action:
                  - bedrock-agentcore:StopRuntimeSession
                Resource:
                  - !Sub "arn:aws:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:runtime/col_ast_orchestrator*"
              - Sid: SSMParameterRead
                Effect: Allow
                Action:
                  - ssm:GetParameter
                  - ssm:GetParameters
                  - ssm:GetParametersByPath
                Resource:
                  - !Sub "arn:aws:ssm:${AWS::Region}:${AWS::AccountId}:parameter/agentcore/*"
              - Sid: SecretsManagerAccess
                Effect: Allow
                Action:
                  - secretsmanager:GetSecretValue
                Resource:
                  - !Sub "arn:aws:secretsmanager:${AWS::Region}:${AWS::AccountId}:secret:*"
              - Sid: KMSDecryptAccess
                Effect: Allow
                Action:
                  - kms:Decrypt
                Resource:
                  - !Sub "arn:aws:kms:${AWS::Region}:${AWS::AccountId}:key/*"
      Tags:
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  # ---------------------------------------------------------------------------
  # Redis Service Catalog product
  # ---------------------------------------------------------------------------
  RedisProvisionedProduct:
    Type: AWS::ServiceCatalog::CloudFormationProvisionedProduct
    DependsOn:
      - RedisAdditionalSecurityGroupParameter
    Properties:
      ProductId: !Ref RedisProductId
      ProvisioningArtifactId: !Ref RedisProvisioningArtifactId
      ProvisionedProductName: !Sub "${AppName}-${Environment}-redis"
      ProvisioningParameters:
        - Key: FriendlyStackName
          Value: !Sub "${AppName}-${Environment}-redis"
        - Key: CacheNodeType
          Value: !Ref RedisCacheNodeType
        - Key: EngineVersion
          Value: !Ref RedisEngineVersion
        - Key: RedisPort
          Value: !Ref RedisPort
        - Key: NumNodeGroups
          Value: !Ref RedisNumNodeGroups
        - Key: ReplicasPerNodeGroup
          Value: !Ref RedisReplicasPerNodeGroup
        - Key: MultiAZEnabled
          Value: !Ref RedisMultiAZEnabled
        - Key: SnapshotRetentionLimit
          Value: !Ref RedisSnapshotRetentionLimit
        - Key: AdditionalSG
          Value: !Ref RedisAdditionalSecurityGroup
        - Key: Subnet1
          Value: !Ref ElastiCacheSubnet1
        - Key: Subnet2
          Value: !Ref ElastiCacheSubnet2
        - Key: Subnet3
          Value: !Ref ElastiCacheSubnet1
        - Key: VpcId
          Value: !Ref VpcIdPath
        - Key: LogFormat
          Value: !Ref RedisLogFormat
        - Key: LogType
          Value: !Ref RedisLogType
        - Key: LogRetentionDays
          Value: !Ref RedisLogRetentionDays
        - Key: CacheLogsEnabled
          Value: !Ref RedisCacheLogsEnabled
      Tags:
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  # ---------------------------------------------------------------------------
  # AgentCore Service Catalog product
  # ---------------------------------------------------------------------------

  AgentCoreLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub "/aws/bedrock-agentcore/col_ast_orchestrator_agentcore_${Environment}"
      RetentionInDays: 90
      Tags:
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  AgentCoreProvisionedProduct:
    Type: AWS::ServiceCatalog::CloudFormationProvisionedProduct
    DependsOn:
      - AgentCoreRuntimeExecutionRole
      - AgentCoreAdditionalSecurityGroupParameter
      - RedisProvisionedProduct
    Properties:
      ProductId: !Ref AgentCoreProductId
      ProvisioningArtifactId: !Ref AgentCoreProvisioningArtifactId
      ProvisionedProductName: !Sub "${AppName}-${Environment}-agentcore"
      ProvisioningParameters:
        - Key: Name
          Value: !Sub "col_ast_orchestrator_agentcore_${Environment}"
        - Key: Description
          Value: !Sub "Colleague Assist Orchestrator AgentCore runtime for ${Environment}"
        - Key: ExecutionRoleArn
          Value: !GetAtt AgentCoreRuntimeExecutionRole.Arn
        - Key: ProtocolConfiguration
          Value: !Ref AgentCoreProtocol
        - Key: CodeS3Bucket
          Value: !Ref LambdaS3BucketName
        - Key: CodeS3Key
          Value: !Ref AgentCoreCodeS3Key
        - Key: CodeRuntime
          Value: !Ref AgentCoreCodeRuntime
        - Key: CodeEntrypoint
          Value: !Ref AgentCoreCodeEntrypoint
        - Key: VpcSubnet1
          Value: !Ref AgentCoreSubnet1
        - Key: VpcSubnet2
          Value: !Ref AgentCoreSubnet2
        - Key: VpcCidr
          Value: !Ref VpcCidrSsmPath
        - Key: AdditionalVpcSecurityGroupId
          Value: !Ref AgentCoreAdditionalSecurityGroup
        - Key: EnvironmentVariables
          Value: !Sub
            - |-
              {"REDIS_URL":"redis://${RedisHost}:${RedisPort}","REDIS_AUTH_SECRET":"${RedisSecretArn}","BU":"${AgentCoreBusinessUnit}","CONFIG_BASE_PATH":"${AgentCoreConfigBasePath}","CONFIG_CACHE_TTL":"${AgentCoreConfigCacheTtl}","LOG_LEVEL":"${AgentCoreLogLevel}"}
            - RedisHost: !GetAtt RedisProvisionedProduct.Outputs.CachePrimaryEndPointAddress
              RedisPort: !GetAtt RedisProvisionedProduct.Outputs.CachePrimaryEndPointPort
              RedisSecretArn: !GetAtt RedisProvisionedProduct.Outputs.CacheSecretArn
      Tags:
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  # ---------------------------------------------------------------------------
  # Lambda IAM role for normal CloudFormation-created Lambda
  # ---------------------------------------------------------------------------
  InteractionStreamLambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub "svc-${AppName}-${Environment}-interaction-stream-lambda-role"
      PermissionsBoundary: !Ref PermissionsBoundaryArn
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
                  - !Sub "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:*"

              - Sid: InvokeAgentRuntime
                Effect: Allow
                Action:
                  - bedrock-agentcore:InvokeAgentRuntime
                Resource:
                  - !Sub "arn:aws:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:runtime/col_ast_orchestrator*"

              - Sid: StopAgentCoreSession
                Effect: Allow
                Action:
                  - bedrock-agentcore:StopRuntimeSession
                Resource:
                  - !Sub "arn:aws:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:runtime/col_ast_orchestrator*"

        - PolicyName: S3ReadPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Sid: S3ListAccess
                Effect: Allow
                Action:
                  - s3:ListBucket
                Resource:
                  - !Sub "arn:aws:s3:::ail-orchestrator-artifacts-*"

              - Sid: S3ObjectReadAccess
                Effect: Allow
                Action:
                  - s3:GetObject
                Resource:
                  - !Sub "arn:aws:s3:::ail-orchestrator-artifacts-*/*"

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

        - PolicyName: SecretsManagerReadPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Sid: SecretsManagerAccess
                Effect: Allow
                Action:
                  - secretsmanager:GetSecretValue
                Resource:
                  - !Sub "arn:aws:secretsmanager:${AWS::Region}:${AWS::AccountId}:secret:*"
      Tags:
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  # ---------------------------------------------------------------------------
  # Normal Lambda function created directly by CloudFormation
  # ---------------------------------------------------------------------------
  InteractionStreamLambda:
    Type: AWS::Lambda::Function
    DependsOn:
      - RedisProvisionedProduct
      - AgentCoreProvisionedProduct
      - LambdaSecurityGroupParameter
      - InteractionStreamLambdaExecutionRole
      - BedrockAgentCoreVpcEndpoint
    Properties:
      FunctionName: !Sub "${AppName}-${Environment}-interaction-stream-lambda"
      Description: !Ref LambdaDescription
      Runtime: !Ref LambdaRuntime
      Handler: !Ref LambdaHandler
      Role: !GetAtt InteractionStreamLambdaExecutionRole.Arn
      Timeout: !Ref LambdaTimeout
      MemorySize: !Ref LambdaMemorySize
      Architectures:
        - !Ref LambdaArchitecture
      TracingConfig:
        Mode: !Ref LambdaTracing
      Code:
        S3Bucket: !Ref LambdaS3BucketName
        S3Key: !Ref LambdaCodeKeyName
      VpcConfig:
        SecurityGroupIds:
          - !Ref LambdaSecurityGroup
        SubnetIds:
          - !Ref LambdaSubnet1
          - !Ref LambdaSubnet2
      Environment:
        Variables:
          AWS_ACCOUNT_ID: !Sub "${AWS::AccountId}"
          AGENTCORE_ENDPOINT: !GetAtt AgentCoreProvisionedProduct.Outputs.RuntimeArn
          AGENTCORE_RUNTIME_ARN: !GetAtt AgentCoreProvisionedProduct.Outputs.RuntimeArn
          CONFIG_S3_BUCKET: !Ref LambdaS3BucketName
          CONNECT_TIMEOUT: !Sub "${ConnectTimeoutSeconds}"
          DEDUP_TTL_SECONDS: !Sub "${DedupTtlSeconds}"
          EVENT_CONFIG_S3_KEY: !Ref EventConfigS3Key
          ORCHESTRATOR_TRIGGER_INTERVAL_SECONDS: !Sub "${OrchestratorTriggerIntervalSeconds}"
          REDIS_AUTH_SECRET: !GetAtt RedisProvisionedProduct.Outputs.CacheSecretArn
          REDIS_HOST: !GetAtt RedisProvisionedProduct.Outputs.CachePrimaryEndPointAddress
          REDIS_PORT: !GetAtt RedisProvisionedProduct.Outputs.CachePrimaryEndPointPort
          REDIS_TLS: !Ref RedisTlsEnabled
          RULES_S3_KEY: !Ref RulesS3Key
          SESSION_TTL_SECONDS: !Sub "${SessionTtlSeconds}"
          SOCKET_TIMEOUT: !Sub "${SocketTimeoutSeconds}"
          STATE_SCHEMA_S3_KEY: !Ref StateSchemaS3Key
          UTTERANCE_THRESHOLD: !Sub "${UtteranceThreshold}"
      Tags:
        - Key: Application
          Value: !Ref AppName
        - Key: Environment
          Value: !Ref Environment

  # ---------------------------------------------------------------------------
  # Connectivity: allow Lambda SG into Redis SG on Redis port
  # ---------------------------------------------------------------------------
  RedisIngressFromLambda:
    Type: AWS::EC2::SecurityGroupIngress
    DependsOn:
      - RedisProvisionedProduct
      - InteractionStreamLambda
    Properties:
      GroupId: !Ref RedisAdditionalSecurityGroup
      IpProtocol: tcp
      FromPort: !Ref RedisPort
      ToPort: !Ref RedisPort
      SourceSecurityGroupId: !Ref LambdaSecurityGroup
      Description: !Sub "Allow Lambda to Redis on port ${RedisPort}"

  RedisIngressFromAgentCore:
    Type: AWS::EC2::SecurityGroupIngress
    DependsOn:
      - RedisProvisionedProduct
      - AgentCoreProvisionedProduct
    Properties:
      GroupId: !Ref RedisAdditionalSecurityGroup
      IpProtocol: tcp
      FromPort: !Ref RedisPort
      ToPort: !Ref RedisPort
      SourceSecurityGroupId: !Ref AgentCoreAdditionalSecurityGroup
      Description: !Sub "Allow AgentCore to Redis on port ${RedisPort}"

  # ---------------------------------------------------------------------------
  # Kinesis trigger on CloudFormation-created Lambda
  # ---------------------------------------------------------------------------
  KinesisEventSourceMapping:
    Type: AWS::Lambda::EventSourceMapping
    DependsOn:
      - InteractionStreamLambda
      - InteractionStreamLambdaExecutionRole
    Properties:
      EventSourceArn: !Ref KinesisStreamArn
      FunctionName: !GetAtt InteractionStreamLambda.Arn
      StartingPosition: !Ref KinesisStartingPosition
      BatchSize: !Ref KinesisBatchSize
      Enabled: true

Outputs:
  RedisAdditionalSecurityGroupIdOutput:
    Description: Redis additional security group ID created by this template.
    Value: !Ref RedisAdditionalSecurityGroup

  LambdaSecurityGroupIdOutput:
    Description: Lambda security group ID created by this template.
    Value: !Ref LambdaSecurityGroup

  AgentCoreAdditionalSecurityGroupIdOutput:
    Description: AgentCore additional security group ID created by this template.
    Value: !Ref AgentCoreAdditionalSecurityGroup

  RedisAdditionalSecurityGroupSsmParameter:
    Description: SSM parameter storing Redis additional security group ID.
    Value: !Ref RedisAdditionalSecurityGroupId

  LambdaSecurityGroupSsmParameter:
    Description: SSM parameter storing Lambda security group ID.
    Value: !Ref LambdaSecurityGroupId

  AgentCoreAdditionalSecurityGroupSsmParameter:
    Description: SSM parameter storing AgentCore additional security group ID.
    Value: !Ref AgentCoreAdditionalSecurityGroupId

  RedisEndpointAddress:
    Description: Redis primary endpoint address from Service Catalog output.
    Value: !GetAtt RedisProvisionedProduct.Outputs.CachePrimaryEndPointAddress

  RedisEndpointPort:
    Description: Redis primary endpoint port from Service Catalog output CachePrimaryEndPointPort.
    Value: !GetAtt RedisProvisionedProduct.Outputs.CachePrimaryEndPointPort

  RedisAuthSecretArn:
    Description: Redis auth secret ARN from Service Catalog output.
    Value: !GetAtt RedisProvisionedProduct.Outputs.CacheSecretArn

  LambdaFunctionArn:
    Description: Lambda function ARN created by this template.
    Value: !GetAtt InteractionStreamLambda.Arn

  AgentCoreExecutionRoleArn:
    Description: AgentCore runtime execution role ARN created by this template.
    Value: !GetAtt AgentCoreRuntimeExecutionRole.Arn

  AgentCoreRuntimeArn:
    Description: AgentCore runtime ARN from the AgentCore Service Catalog product.
    Value: !GetAtt AgentCoreProvisionedProduct.Outputs.RuntimeArn

  BedrockAgentCoreVpcEndpointId:
    Description: VPC Interface Endpoint ID for bedrock-agentcore (PrivateLink), required since this VPC has no NAT Gateway.
    Value: !Ref BedrockAgentCoreVpcEndpoint
