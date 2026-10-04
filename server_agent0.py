import argparse
import base64
import logging
import os
import pickle

from flask import Flask
from flask import jsonify
from flask import request

from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量保存加载的模型
policy = None


def load_model(config_name: str, checkpoint_dir: str, pytorch_device: str | None = None):
    """加载模型（支持 JAX 和 PyTorch 权重）
    
    Args:
        config_name: 配置名称
        checkpoint_dir: checkpoint 目录路径，如果包含 model.safetensors 则自动使用 PyTorch 加载
        pytorch_device: PyTorch 设备（如 "cuda", "cpu", "cuda:0"），如果为 None 则自动检测
    """
    global policy
    try:
        logger.info("开始加载模型...")
        config = _config.get_config(config_name)

        # 检查是否是 PyTorch 模型（通过检查 model.safetensors 文件）
        weight_path = os.path.join(checkpoint_dir, "model.safetensors")
        is_pytorch = os.path.exists(weight_path)
        
        if is_pytorch:
            logger.info(f"检测到 PyTorch 权重文件: {weight_path}")
            logger.info("将使用 PyTorch 模型加载")
        else:
            logger.info("未检测到 PyTorch 权重文件，将使用 JAX 模型加载")
            logger.info(f"请确保 checkpoint_dir 包含 model.safetensors 文件以使用 PyTorch 权重")

        # 创建训练好的策略（会自动检测并使用 PyTorch 或 JAX）
        policy = _policy_config.create_trained_policy(
            config, 
            checkpoint_dir,
            pytorch_device=pytorch_device
        )
        logger.info("模型加载成功!")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "healthy", "model_loaded": policy is not None})


@app.route("/infer", methods=["POST"])
def infer():
    """推理接口"""
    global policy

    if policy is None:
        return jsonify({"error": "模型未加载"}), 500

    try:
        # 获取请求数据
        data = request.get_json()

        if "example" not in data:
            return jsonify({"error": "缺少example数据"}), 400

        # 解码example数据 (假设客户端会发送base64编码的pickle数据)
        example_b64 = data["example"]
        example_bytes = base64.b64decode(example_b64)
        example = pickle.loads(example_bytes)
        example["prompt"] = example["task"]

        logger.info("收到推理请求")

        # 运行推理
        result = policy.infer(example)

        # 将结果序列化
        result_bytes = pickle.dumps(result)
        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        logger.info("推理完成")

        return jsonify(
            {
                "success": True,
                "result": result_b64,
                "actions_shape": list(result["actions"].shape) if "actions" in result else None,
            }
        )

    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        return jsonify({"error": f"推理失败: {e!s}"}), 500


@app.route("/test", methods=["GET"])
def test():
    """测试接口，使用示例数据进行推理"""
    global policy

    if policy is None:
        return jsonify({"error": "模型未加载"}), 500

    try:
        # 创建示例数据
        example = droid_policy.make_droid_example()

        # 运行推理
        result = policy.infer(example)

        return jsonify(
            {
                "success": True,
                "message": "测试推理成功",
                "actions_shape": list(result["actions"].shape) if "actions" in result else None,
            }
        )

    except Exception as e:
        logger.error(f"测试推理出错: {e}")
        return jsonify({"error": f"测试失败: {e!s}"}), 500


if __name__ == "__main__":
    logger.info("启动服务器...")

    parser = argparse.ArgumentParser(description="PI0 Agent0 inference server (支持 JAX 和 PyTorch 权重)")
    parser.add_argument(
        "--config", type=str, default="pi0_fast_droid_train_lora_agent0", help="Config name for model setup"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True, 
        help="Path to checkpoint directory. 如果包含 model.safetensors 文件，将自动使用 PyTorch 权重"
    )
    parser.add_argument(
        "--pytorch_device",
        type=str,
        default=None,
        help="PyTorch 设备 (如 'cuda', 'cpu', 'cuda:0')。如果为 None，将自动检测（优先使用 cuda）"
    )
    parser.add_argument("--port", type=int, default=20019, help="Port for the Flask server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use a trusted robot network")
    args = parser.parse_args()

    # 启动时加载模型
    if not load_model(args.config, args.checkpoint_dir, pytorch_device=args.pytorch_device):
        logger.error("无法启动服务器: 模型加载失败")
        exit(1)

    # 启动Flask服务器
    app.run(host=args.host, port=args.port, debug=False)
