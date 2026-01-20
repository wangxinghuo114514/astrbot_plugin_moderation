"""
AstrBot AI消息审核插件

功能：
- 基于AI的文字和图片消息审核
- 自动撤回违规消息并禁言
- 上下文分析减少误判
- AI学习管理员反馈
- 刷屏检测和踢人
- WebUI管理界面

作者：wangxinghuo
版本：1.0.0
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, deque
from aiohttp import web
import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.provider import LLMResponse
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from pathlib import Path


@register("moderation", "iFlow", "AI消息审核插件", "1.0.0")
class ModerationPlugin(Star):
    def __init__(self, context: Context, config):
        super().__init__(context)
        self.config = config
        self.name = "astrbot_plugin_moderation"

        # 消息上下文存储 {group_id: deque of messages}
        self.message_contexts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

        # 审核记录存储
        self.moderation_records: List[Dict] = []
        self._load_moderation_records()

        # 学习数据存储
        self.learning_data: Dict[str, List[Dict]] = {"correct": [], "incorrect": []}
        self._load_learning_data()

        # WebUI服务器
        self.webui_app = None
        self.webui_runner = None
        self.webui_site = None
        self._start_webui_server()

        # 刷屏检测：记录用户违规历史 {(group_id, sender_id): {"count": int, "first_time": datetime, "last_time": datetime}}
        self.spam_violations: Dict[tuple, Dict] = {}

        # 合并违规记录：记录待合并的违规 {(group_id, sender_id): [violation_records]}
        self.pending_violations: Dict[tuple, List[Dict]] = {}

        logger.info("AI消息审核插件已加载")

    def _load_moderation_records(self):
        """加载审核记录"""
        try:
            data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
            data_path.mkdir(parents=True, exist_ok=True)
            records_file = data_path / "moderation_records.json"
            if records_file.exists():
                with open(records_file, "r", encoding="utf-8") as f:
                    self.moderation_records = json.load(f)
        except Exception as e:
            logger.error(f"加载审核记录失败: {e}")

    def _save_moderation_records(self):
        """保存审核记录"""
        try:
            data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
            data_path.mkdir(parents=True, exist_ok=True)
            records_file = data_path / "moderation_records.json"

            # 清理无法序列化的数据
            clean_records = []
            for record in self.moderation_records:
                try:
                    # 深度复制并清理数据
                    clean_record = {
                        "id": str(record.get("id", "")),
                        "group_id": str(record.get("group_id", "")),
                        "sender_id": str(record.get("sender_id", "")),
                        "sender_name": str(record.get("sender_name", "")),
                        "message": str(record.get("message", "")),
                        "message_type": str(record.get("message_type", "")),
                        "context": record.get("context", []),
                        "ai_result": record.get("ai_result", {}),
                        "is_violation": bool(record.get("is_violation", False)),
                        "reason": str(record.get("reason", "")),
                        "severity": int(record.get("severity", 3)),
                        "timestamp": str(record.get("timestamp", "")),
                        "admin_reviewed": bool(record.get("admin_reviewed", False)),
                        "admin_correct": record.get("admin_correct"),
                        "admin_comment": str(record.get("admin_comment", "")),
                    }
                    clean_records.append(clean_record)
                except Exception as e:
                    logger.warning(f"清理记录数据失败: {e}, 跳过此记录")
                    continue

            with open(records_file, "w", encoding="utf-8") as f:
                json.dump(clean_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存审核记录失败: {e}")

    def _load_learning_data(self):
        """加载学习数据"""
        try:
            data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
            data_path.mkdir(parents=True, exist_ok=True)
            learning_file = data_path / "learning_data.json"
            if learning_file.exists():
                with open(learning_file, "r", encoding="utf-8") as f:
                    self.learning_data = json.load(f)
        except Exception as e:
            logger.error(f"加载学习数据失败: {e}")

    def _save_learning_data(self):
        """保存学习数据"""
        try:
            data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
            data_path.mkdir(parents=True, exist_ok=True)
            learning_file = data_path / "learning_data.json"
            with open(learning_file, "w", encoding="utf-8") as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存学习数据失败: {e}")

    def _start_webui_server(self):
        """启动独立的WebUI服务器"""
        try:
            self.webui_app = web.Application()
            self._setup_webui_routes()

            port = self.config.get("webui_port", 8899)
            self.webui_runner = web.AppRunner(self.webui_app)
            asyncio.create_task(self._run_webui_server(port))

            logger.info(f"WebUI服务器启动在端口 {port}")
        except Exception as e:
            logger.error(f"启动WebUI服务器失败: {e}")

    async def _run_webui_server(self, port: int):
        """运行WebUI服务器"""
        try:
            await self.webui_runner.setup()
            self.webui_site = web.TCPSite(self.webui_runner, "0.0.0.0", port)
            await self.webui_site.start()
            logger.info(f"WebUI服务器已启动，访问地址: http://localhost:{port}")
        except Exception as e:
            logger.error(f"运行WebUI服务器失败: {e}")

    def _setup_webui_routes(self):
        """设置WebUI路由"""
        webui_dir = Path(__file__).parent / "webui"

        # 登录页面
        async def login_page(request):
            login_file = webui_dir / "login.html"
            if login_file.exists():
                with open(login_file, "r", encoding="utf-8") as f:
                    return web.Response(text=f.read(), content_type="text/html")
            return web.Response(text="登录页面不存在", status=404)

        # 登录API
        async def login_api(request):
            try:
                data = await request.json()
                password = data.get("password", "")
                correct_password = self.config.get("webui_password", "admin123")

                if password == correct_password:
                    return web.json_response({"success": True, "message": "登录成功"})
                else:
                    return web.json_response({"success": False, "message": "密码错误"})
            except Exception as e:
                logger.error(f"登录失败: {e}")
                return web.json_response({"success": False, "message": str(e)})

        # 主页面
        async def index_page(request):
            index_file = webui_dir / "index.html"
            if index_file.exists():
                with open(index_file, "r", encoding="utf-8") as f:
                    return web.Response(text=f.read(), content_type="text/html")
            return web.Response(text="WebUI文件不存在", status=404)

        # API路由：获取审核记录
        async def api_get_records(request):
            return web.json_response({"records": self.moderation_records[-100:]})

        # API路由：获取学习数据
        async def api_get_learning(request):
            return web.json_response(self.learning_data)

        # API路由：提交管理员反馈
        async def api_submit_feedback(request):
            try:
                data = await request.json()
                record_id = data.get("record_id")
                is_correct = data.get("is_correct")
                admin_comment = data.get("comment", "")

                # 查找对应的审核记录
                record = None
                for r in self.moderation_records:
                    if r.get("id") == record_id:
                        record = r
                        break

                if record:
                    # 更新记录
                    record["admin_reviewed"] = True
                    record["admin_correct"] = is_correct
                    record["admin_comment"] = admin_comment
                    record["review_time"] = datetime.now().isoformat()

                    # 添加到学习数据
                    learning_item = {
                        "message": record.get("message"),
                        "context": record.get("context"),
                        "ai_result": record.get("ai_result"),
                        "admin_judgment": is_correct,
                        "comment": admin_comment,
                        "timestamp": datetime.now().isoformat(),
                    }

                    if is_correct:
                        self.learning_data["correct"].append(learning_item)
                    else:
                        self.learning_data["incorrect"].append(learning_item)

                    self._save_moderation_records()
                    self._save_learning_data()

                    # 检查是否需要更新AI学习
                    if self.config.get("learning_enabled", True):
                        threshold = self.config.get("learning_threshold", 10)
                        if (
                            len(self.learning_data["correct"]) >= threshold
                            or len(self.learning_data["incorrect"]) >= threshold
                        ):
                            await self._update_ai_learning()

                    return web.json_response({"success": True, "message": "反馈已提交"})
                else:
                    return web.json_response(
                        {"success": False, "message": "记录不存在"}
                    )
            except Exception as e:
                logger.error(f"提交反馈失败: {e}")
                return web.json_response({"success": False, "message": str(e)})

        # API路由：获取统计信息
        async def api_get_stats(request):
            total = len(self.moderation_records)
            violations = sum(
                1 for r in self.moderation_records if r.get("is_violation")
            )
            reviewed = sum(
                1 for r in self.moderation_records if r.get("admin_reviewed")
            )
            correct = sum(1 for r in self.moderation_records if r.get("admin_correct"))

            return web.json_response(
                {
                    "total": total,
                    "violations": violations,
                    "reviewed": reviewed,
                    "accuracy": round(correct / reviewed * 100, 2)
                    if reviewed > 0
                    else 0,
                    "learning_samples": {
                        "correct": len(self.learning_data["correct"]),
                        "incorrect": len(self.learning_data["incorrect"]),
                    },
                }
            )

        # API路由：代理图片
        async def api_proxy_image(request):
            """代理图片请求，解决跨域问题"""
            try:
                image_url = request.query.get("url")
                if not image_url:
                    return web.Response(status=400, text="缺少url参数")

                import aiohttp

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        image_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            # 获取Content-Type
                            content_type = response.headers.get(
                                "Content-Type", "image/jpeg"
                            )
                            return web.Response(
                                body=image_data, content_type=content_type
                            )
                        else:
                            return web.Response(
                                status=response.status,
                                text=f"图片下载失败: {response.status}",
                            )
            except Exception as e:
                logger.error(f"代理图片失败: {e}")
                return web.Response(status=500, text=f"代理图片失败: {str(e)}")

        # API路由：删除学习样本
        async def api_delete_learning_sample(request):
            """删除学习样本"""
            try:
                data = json.loads(request.body)
                sample_type = data.get("type")  # "correct" or "incorrect"
                sample_index = data.get("index")

                if sample_type not in ["correct", "incorrect"]:
                    return web.json_response(
                        {"success": False, "message": "无效的样本类型"}
                    )

                if sample_index is None or sample_index < 0:
                    return web.json_response(
                        {"success": False, "message": "无效的样本索引"}
                    )

                # 删除样本
                if sample_type == "correct":
                    if 0 <= sample_index < len(self.learning_data["correct"]):
                        del self.learning_data["correct"][sample_index]
                else:
                    if 0 <= sample_index < len(self.learning_data["incorrect"]):
                        del self.learning_data["incorrect"][sample_index]

                self._save_learning_data()
                return web.json_response({"success": True, "message": "样本已删除"})
            except Exception as e:
                logger.error(f"删除学习样本失败: {e}")
                return web.json_response({"success": False, "message": str(e)})

        # API路由：编辑学习样本
        async def api_edit_learning_sample(request):
            """编辑学习样本"""
            try:
                data = json.loads(request.body)
                sample_type = data.get("type")  # "correct" or "incorrect"
                sample_index = data.get("index")
                new_comment = data.get("comment", "")

                if sample_type not in ["correct", "incorrect"]:
                    return web.json_response(
                        {"success": False, "message": "无效的样本类型"}
                    )

                if sample_index is None or sample_index < 0:
                    return web.json_response(
                        {"success": False, "message": "无效的样本索引"}
                    )

                # 编辑样本
                if sample_type == "correct":
                    if 0 <= sample_index < len(self.learning_data["correct"]):
                        self.learning_data["correct"][sample_index]["comment"] = (
                            new_comment
                        )
                else:
                    if 0 <= sample_index < len(self.learning_data["incorrect"]):
                        self.learning_data["incorrect"][sample_index]["comment"] = (
                            new_comment
                        )

                self._save_learning_data()
                return web.json_response({"success": True, "message": "样本已更新"})
            except Exception as e:
                logger.error(f"编辑学习样本失败: {e}")
                return web.json_response({"success": False, "message": str(e)})

        # 注册路由
        self.webui_app.router.add_get("/", login_page)
        self.webui_app.router.add_post("/api/login", login_api)
        self.webui_app.router.add_get("/index", index_page)
        self.webui_app.router.add_get("/api/records", api_get_records)
        self.webui_app.router.add_get("/api/learning", api_get_learning)
        self.webui_app.router.add_post("/api/feedback", api_submit_feedback)
        self.webui_app.router.add_get("/api/stats", api_get_stats)
        self.webui_app.router.add_get("/api/proxy/image", api_proxy_image)
        self.webui_app.router.add_post(
            "/api/learning/delete", api_delete_learning_sample
        )
        self.webui_app.router.add_post("/api/learning/edit", api_edit_learning_sample)

        logger.info("WebUI路由设置完成")

    async def _update_ai_learning(self):
        """更新AI学习数据"""
        try:
            # 构建学习提示词
            correct_samples = self.learning_data["correct"][-10:]
            incorrect_samples = self.learning_data["incorrect"][-10:]

            learning_prompt = "以下是一些审核案例，请学习这些案例以提高审核准确性：\n\n"

            if correct_samples:
                learning_prompt += "正确的审核案例：\n"
                for i, sample in enumerate(correct_samples[-5:], 1):
                    learning_prompt += f"{i}. 消息：{sample['message']}\n"
                    learning_prompt += f"   AI判断：{sample['ai_result']}\n"
                    learning_prompt += f"   管理员确认：正确\n\n"

            if incorrect_samples:
                learning_prompt += "错误的审核案例：\n"
                for i, sample in enumerate(incorrect_samples[-5:], 1):
                    learning_prompt += f"{i}. 消息：{sample['message']}\n"
                    learning_prompt += f"   AI判断：{sample['ai_result']}\n"
                    learning_prompt += f"   管理员确认：错误\n"
                    if sample.get("comment"):
                        learning_prompt += f"   管理员备注：{sample['comment']}\n\n"

            learning_prompt += "请根据以上案例，调整你的审核标准，减少误判。"

            # 保存学习提示词到文件
            data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
            data_path.mkdir(parents=True, exist_ok=True)
            learning_file = data_path / "learning_prompt.txt"
            with open(learning_file, "w", encoding="utf-8") as f:
                f.write(learning_prompt)

            logger.info("AI学习数据已更新")
        except Exception as e:
            logger.error(f"更新AI学习数据失败: {e}")

    def _should_moderate(self, event: AstrMessageEvent) -> bool:
        """检查是否应该审核该消息"""
        group_id = event.get_group_id()
        sender_id = event.get_sender_id()

        # 检查用户白名单（免审核用户）
        whitelist_users = self.config.get("whitelist_users", [])
        if sender_id in whitelist_users:
            logger.info(f"用户 {sender_id} 在免审核列表中，跳过审核")
            return False

        # 检查群聊是否在审核白名单中
        whitelist_groups = self.config.get("whitelist_groups", [])
        if not whitelist_groups:
            logger.debug("未配置审核群聊白名单，不进行审核")
            return False

        if group_id not in whitelist_groups:
            return False

        return True  # 在白名单中，需要审核

    def _check_local_keywords(self, message: str) -> Dict:
        """本地关键词快速检查"""
        # 明显的违规关键词列表（只保留二字及以上的明确违规词）
        violation_keywords = {
            "鸡巴": {"severity": 5, "reason": "色情词汇"},
            "阴茎": {"severity": 5, "reason": "色情词汇"},
            "性交": {"severity": 5, "reason": "色情词汇"},
            "做爱": {"severity": 5, "reason": "色情词汇"},
            "几把": {"severity": 5, "reason": "色情词汇"},
            "草饲": {"severity": 4, "reason": "侮辱性词汇"},
            "操你妈": {"severity": 4, "reason": "严重辱骂"},
            "傻逼": {"severity": 4, "reason": "侮辱性词汇"},
            "妈的": {"severity": 3, "reason": "不文明用语"},
            "死妈": {"severity": 4, "reason": "恶意辱骂"},
            "操你": {"severity": 4, "reason": "侮辱性词汇"},
            "草泥马": {"severity": 4, "reason": "侮辱性词汇"},
            "他妈": {"severity": 3, "reason": "不文明用语"},
            "傻X": {"severity": 4, "reason": "侮辱性词汇"},
            "傻B": {"severity": 4, "reason": "侮辱性词汇"},
            "傻比": {"severity": 4, "reason": "侮辱性词汇"},
            "傻逼": {"severity": 4, "reason": "侮辱性词汇"},
            "废物": {"severity": 3, "reason": "侮辱性词汇"},
            "垃圾": {"severity": 3, "reason": "侮辱性词汇"},
        }

        for keyword, info in violation_keywords.items():
            if keyword in message:
                logger.info(
                    f"本地关键词检测到违规: {keyword}, 严重程度: {info['severity']}"
                )
                return {
                    "is_violation": True,
                    "reason": f"包含违规关键词：{keyword}（{info['reason']}）",
                    "severity": info["severity"],
                    "detected_by": "local_keywords",
                }

        return {"is_violation": False}

    def _store_message_context(self, event: AstrMessageEvent):
        """存储消息上下文"""
        group_id = event.get_group_id()
        if not group_id:
            return

        message_data = {
            "sender_id": event.get_sender_id(),
            "sender_name": event.get_sender_name(),
            "message": event.message_str,
            "timestamp": datetime.now().isoformat(),
            "message_id": event.message_obj.message_id,
        }

        self.message_contexts[group_id].append(message_data)

    def _get_message_context(
        self, event: AstrMessageEvent, count: int = 5
    ) -> List[Dict]:
        """获取消息上下文"""
        group_id = event.get_group_id()
        if not group_id:
            return []

        context = list(self.message_contexts[group_id])
        # 排除当前消息
        current_message_id = event.message_obj.message_id
        context = [
            msg for msg in context if msg.get("message_id") != current_message_id
        ]

        return context[-count:] if context else []

    async def _download_image_to_base64(self, image_url: str) -> Optional[str]:
        """下载图片并转换为base64"""
        try:
            import aiohttp

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    image_url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        logger.info(f"成功下载图片，大小: {len(image_data)} bytes")
                        import base64

                        return base64.b64encode(image_data).decode("utf-8")
                    else:
                        logger.warning(f"图片下载失败，状态码: {response.status}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"下载图片超时: {image_url}")
            return None
        except Exception as e:
            logger.error(f"下载图片失败: {e}, URL: {image_url}")
            return None

    async def _extract_text_from_image(self, image_data: bytes) -> Optional[str]:
        """从图片中提取文字（OCR）"""
        try:
            import io
            from PIL import Image
            import pytesseract

            # 将字节数据转换为PIL Image
            image = Image.open(io.BytesIO(image_data))

            # 使用OCR提取文字
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")

            logger.info(
                f"OCR提取文字成功: {text[:100]}..."
                if len(text) > 100
                else f"OCR提取文字成功: {text}"
            )
            return text.strip()
        except ImportError:
            logger.warning("未安装pytesseract或PIL库，无法进行OCR识别")
            return None
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return None

    async def _extract_text_from_image_url(self, image_url: str) -> Optional[str]:
        """从图片URL下载并提取文字"""
        try:
            import aiohttp

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    image_url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        return await self._extract_text_from_image(image_data)
            return None
        except Exception as e:
            logger.warning(f"OCR识别图片URL失败: {e}")
            return None

    async def _moderate_image_with_retry(
        self, event: AstrMessageEvent, image_url: str
    ) -> Dict:
        """带重试机制的图片审核"""
        moderation_result = None
        for retry in range(3):
            try:
                moderation_result = await self._moderate_message(
                    event, image_url, is_image=True
                )
                if moderation_result and moderation_result.get("is_violation"):
                    logger.info(f"图片审核成功: 检测到违规")
                    break
                elif moderation_result and not moderation_result.get("is_violation"):
                    logger.info(f"图片审核成功: 未检测到违规")
                    break
                else:
                    logger.warning(f"图片审核返回无效结果，重试 {retry + 1}/3")
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"图片审核失败 (重试 {retry + 1}/3): {e}")
                await asyncio.sleep(1)

        # 如果审核失败，记录警告但不阻止
        if not moderation_result:
            logger.warning(f"图片审核失败，无法判定: {image_url}")
            moderation_result = {
                "is_violation": False,
                "reason": "图片审核失败，无法判定",
                "severity": 0,
            }

        return moderation_result

    async def _moderate_message(
        self, event: AstrMessageEvent, message_content: str, is_image: bool = False
    ) -> Dict:
        """使用AI审核消息"""
        try:
            # 首先进行本地关键词快速检查
            if not is_image:
                local_result = self._check_local_keywords(message_content)
                if local_result.get("is_violation"):
                    logger.info(f"本地关键词检测到违规，跳过AI审核")
                    return local_result

            # 获取当前发送者信息
            current_sender = event.get_sender_name()
            current_sender_id = event.get_sender_id()

            # 获取上下文
            context_count = self.config.get("context_count", 5)
            context_messages = self._get_message_context(event, context_count)

            # 构建审核提示词
            system_prompt = self.config.get("moderation_system_prompt", "")

            # 添加上下文信息 - 明确标注谁说了什么
            context_text = ""
            if context_messages:
                context_text = "\n\n【上下文消息（最近{}条）】\n".format(
                    len(context_messages)
                )
                for i, ctx_msg in enumerate(context_messages, 1):
                    sender = ctx_msg.get("sender_name", "未知用户")
                    msg = ctx_msg.get("message", "")
                    context_text += f"{i}. 【{sender}】说：{msg}\n"
                context_text += "\n【重要提示：请仔细分析上述上下文，判断当前消息是否在玩梗、开玩笑或正常对话】\n"
                context_text += "\n【当前消息】\n"

            # 构建用户提示词
            if is_image:
                # 图片审核 - AI会自动接收图片数据
                user_prompt = f"{context_text}请审核【{current_sender}】发送的图片是否违规。\n\n注意：\n1. 只审核【{current_sender}】发送的图片，不要将上下文中其他人的消息误判为【{current_sender}】的\n2. 即使是表情包或玩梗，如果图片内容包含明显的违规内容（如色情、暴力、辱骂等），也应判定为违规\n3. 仔细分析图片内容，包括图片中的文字、图案、表情等\n4. 表情包不是违规的挡箭牌，如果表情包本身包含违规内容，必须判定为违规\n5. 如果图片包含文字，请识别文字内容并判断是否违规"
            else:
                user_prompt = f'{context_text}请审核【{current_sender}】发送的文字是否违规。\n\n消息内容：{message_content}\n\n注意：\n1. 只审核【{current_sender}】发送的消息，不要将上下文中其他人的消息误判为【{current_sender}】的\n2. **必须仔细分析上下文**：结合上下文判断当前消息是否在玩梗、开玩笑或正常对话\n3. **区分玩梗和违规**：\n   - 如果是网络流行语、玩梗（如"我丢你老冯"等），且上下文显示是正常对话，不要判定为违规\n   - 如果是明显的侮辱性词汇（如"鸡巴"、"几把"、"操你妈"等），即使上下文在玩梗，也应判定为违规\n4. reason字段中不要复述具体的违规内容，只说明违规类型'

            # 获取AI提供商
            provider_id = self.config.get("ai_provider", "")
            if not provider_id:
                logger.error("未配置AI提供商")
                return {"is_violation": False, "reason": "AI提供商未配置"}

            # 调用AI
            if is_image:
                # 使用多模态API，传递图片URL
                llm_resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    image_urls=[message_content],  # 传递图片URL列表
                )
            else:
                # 文字消息，不传递图片
                llm_resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                )

            # 解析AI响应
            response_text = llm_resp.completion_text.strip()
            logger.info(f"AI审核响应: {response_text[:200]}...")  # 记录AI响应

            try:
                # 尝试解析JSON响应
                result = json.loads(response_text)
                logger.info(
                    f"解析JSON成功: is_violation={result.get('is_violation')}, severity={result.get('severity')}"
                )
            except json.JSONDecodeError:
                # 如果不是JSON格式，尝试从文本中提取信息
                logger.warning(f"JSON解析失败，尝试从文本提取信息")
                result = {
                    "is_violation": "违规" in response_text
                    or "violation" in response_text.lower()
                    or "色情" in response_text
                    or "辱骂" in response_text,
                    "reason": response_text,
                    "severity": 3,
                }
                logger.info(f"文本提取结果: is_violation={result.get('is_violation')}")

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"AI审核失败: {e}")

            # 检查是否是内容安全检查拦截的400错误
            if "data_inspection_failed" in error_msg or "400" in error_msg:
                logger.warning("AI提供商的内容安全检查拦截了请求")

                # 检查消息长度，如果消息太短（如"？"、"？"等），很可能是误判
                if is_image:
                    message_length = len(message_content)  # 图片URL长度
                else:
                    message_length = len(message_content)

                # 如果消息长度小于3，很可能是误判，不判定为违规
                if message_length < 3:
                    logger.warning(
                        f"消息长度过短({message_length}字符)，可能是误判，不判定为违规"
                    )
                    return {
                        "is_violation": False,
                        "reason": f"AI审核失败，消息过短无法准确判断",
                        "severity": 0,
                        "error_type": "safety_check_false_positive",
                    }

                # 检查是否启用安全检查拦截判定
                enable_safety_check_violation = self.config.get(
                    "enable_safety_check_violation", False
                )

                if not enable_safety_check_violation:
                    logger.info("安全检查拦截判定已禁用，不判定为违规")
                    return {
                        "is_violation": False,
                        "reason": f"AI审核失败: 内容安全检查拦截",
                        "severity": 0,
                        "error_type": "safety_check",
                    }

                # 如果启用了安全检查拦截判定，且消息长度足够，才判定为违规
                logger.warning("内容触发了AI提供商的安全检查，判定为违规")
                return {
                    "is_violation": True,
                    "reason": "内容触发了安全检查，包含敏感或违规词汇",
                    "severity": 4,
                    "error_type": "safety_check",
                }

            # 检查是否是网络错误
            if (
                "timeout" in error_msg.lower()
                or "connection" in error_msg.lower()
                or "network" in error_msg.lower()
            ):
                logger.warning("检测到网络错误")
                return {
                    "is_violation": False,
                    "reason": f"网络错误，无法审核: {error_msg}",
                    "severity": 0,
                    "error_type": "network_error",
                }

            return {
                "is_violation": False,
                "reason": f"审核失败: {error_msg}",
                "error_type": "unknown",
            }

    async def _revoke_message(self, event: AstrMessageEvent):
        """撤回消息"""
        try:
            if event.get_platform_name() == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                    AiocqhttpMessageEvent,
                )

                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    payloads = {
                        "message_id": event.message_obj.message_id,
                    }
                    await client.api.call_action("delete_msg", **payloads)
                    logger.info(f"已撤回消息: {event.message_obj.message_id}")
        except Exception as e:
            logger.error(f"撤回消息失败: {e}")

    async def _mute_user(self, event: AstrMessageEvent, duration: int):
        """禁言用户"""
        try:
            if event.get_platform_name() == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                    AiocqhttpMessageEvent,
                )

                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    payloads = {
                        "group_id": event.get_group_id(),
                        "user_id": event.get_sender_id(),
                        "duration": duration * 60,  # 转换为秒
                    }
                    await client.api.call_action("set_group_ban", **payloads)
                    logger.info(f"已禁言用户 {event.get_sender_id()} {duration}分钟")
        except Exception as e:
            logger.error(f"禁言用户失败: {e}")

    async def _kick_user(self, event: AstrMessageEvent):
        """踢出用户"""
        try:
            if event.get_platform_name() == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                    AiocqhttpMessageEvent,
                )

                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    payloads = {
                        "group_id": event.get_group_id(),
                        "user_id": event.get_sender_id(),
                    }
                    await client.api.call_action("set_group_kick", **payloads)
                    logger.info(f"已踢出用户 {event.get_sender_id()}")
        except Exception as e:
            logger.error(f"踢出用户失败: {e}")

    def _sanitize_reason(self, reason: str) -> str:
        """清理违规原因，脱敏处理敏感词汇，并格式化输出"""
        if not reason:
            return "违反群规"

        # 违规类型映射表
        violation_types = {
            "色情": {"type": "色情内容", "rule": "禁止发送色情、性暗示、低俗内容"},
            "性交": {"type": "色情内容", "rule": "禁止发送色情、性行为描述"},
            "阴茎": {"type": "色情内容", "rule": "禁止发送色情、性器官描述"},
            "生殖": {"type": "色情内容", "rule": "禁止发送色情、性暗示内容"},
            "辱骂": {"type": "恶意攻击", "rule": "禁止进行人身攻击、辱骂"},
            "侮辱": {"type": "恶意攻击", "rule": "禁止进行人身攻击、侮辱"},
            "攻击": {"type": "恶意攻击", "rule": "禁止进行人身攻击、恶意贬低"},
            "暴力": {"type": "暴力内容", "rule": "禁止发送暴力、威胁内容"},
            "杀": {"type": "暴力内容", "rule": "禁止发送暴力、威胁内容"},
            "死": {"type": "暴力内容", "rule": "禁止发送暴力、威胁内容"},
            "政治": {"type": "政治敏感", "rule": "禁止发送政治敏感内容"},
            "敏感": {"type": "敏感内容", "rule": "禁止发送敏感、违规内容"},
            "低俗": {"type": "低俗内容", "rule": "禁止发送低俗、不良趣味内容"},
            "违规": {"type": "违规内容", "rule": "违反群规"},
        }

        # 敏感词汇映射表：敏感词 -> 脱敏词
        sensitive_words_map = {
            # 性相关词汇
            "生殖系统": "敏感解剖术语",
            "性交": "敏感行为",
            "阴茎": "敏感器官",
            "阴道": "敏感器官",
            "睾丸": "敏感器官",
            "卵巢": "敏感器官",
            "输卵管": "敏感器官",
            "子宫": "敏感器官",
            "排卵": "敏感生理过程",
            "射精": "敏感生理过程",
            "射精口": "敏感部位",
            "精液": "敏感体液",
            "精子": "敏感细胞",
            "卵子": "敏感细胞",
            "性器官": "敏感器官",
            "性欲": "敏感需求",
            "性行为": "敏感行为",
            "性暗示": "敏感暗示",
            "做爱": "敏感行为",
            "鸡巴": "敏感器官",
            "几把": "敏感器官",
            "阴蒂": "敏感器官",
            "阴唇": "敏感器官",
            "阴毛": "敏感部位",
            "阳具": "敏感器官",
            # 暴力相关词汇
            "杀": "暴力行为",
            "死": "暴力词汇",
            "血": "暴力词汇",
            "刀": "危险物品",
            "枪": "危险物品",
            "炸": "危险行为",
            "毒": "危险物品",
            "砍": "暴力行为",
            "捅": "暴力行为",
            "射": "暴力行为",
            # 侮辱性词汇
            "操": "侮辱性词汇",
            "妈": "侮辱性词汇",
            "逼": "侮辱性词汇",
            "傻": "侮辱性词汇",
            "废物": "侮辱性词汇",
            "垃圾": "侮辱性词汇",
            "畜生": "侮辱性词汇",
            "猪": "侮辱性词汇",
            "狗": "侮辱性词汇",
            # 政治敏感词
            "文革": "敏感历史事件",
            "中央": "敏感机构",
            "政治": "敏感话题",
            "政府": "敏感机构",
            "党": "敏感组织",
            "领导人": "敏感人物",
            # 其他敏感词
            "毒品": "违法物品",
            "赌博": "违法活动",
            "诈骗": "违法活动",
            "色情": "敏感内容",
            "淫秽": "敏感内容",
            "低俗": "低俗内容",
        }

        # 脱敏处理：替换敏感词汇
        sanitized = reason
        for sensitive_word, replacement in sensitive_words_map.items():
            sanitized = sanitized.replace(sensitive_word, replacement)

        # 检测违规类型和违反条款
        violation_type = "违规内容"
        violated_rule = "违反群规"

        for keyword, info in violation_types.items():
            if keyword in sanitized:
                violation_type = info["type"]
                violated_rule = info["rule"]
                break

        # 构建结构化的违规原因
        formatted_reason = f"【违规类型】{violation_type}\n"
        formatted_reason += f"【违反条款】{violated_rule}\n"

        # 提取违规分析（取reason的前80字作为分析）
        analysis = sanitized[:80] + "..." if len(sanitized) > 80 else sanitized
        formatted_reason += f"【违规分析】{analysis}"

        return formatted_reason

    def _should_merge_record(self, record: Dict) -> bool:
        """检查是否应该合并记录（短时间内的重复违规）"""
        # 检查是否启用记录合并
        if not self.config.get("enable_record_merge", True):
            return False

        # 只合并违规记录
        if not record.get("is_violation"):
            return False

        group_id = record["group_id"]
        sender_id = record["sender_id"]
        current_time = datetime.fromisoformat(record["timestamp"])

        # 查找最近的同一用户的违规记录
        for existing_record in reversed(self.moderation_records):
            if (
                existing_record.get("group_id") == group_id
                and existing_record.get("sender_id") == sender_id
                and existing_record.get("is_violation")
            ):
                existing_time = datetime.fromisoformat(existing_record["timestamp"])
                time_diff = (current_time - existing_time).total_seconds()

                # 30秒内的同一用户违规记录合并
                if time_diff <= 30:
                    return True
                else:
                    break

        return False

    def _merge_record(self, new_record: Dict) -> Dict:
        """合并违规记录"""
        group_id = new_record["group_id"]
        sender_id = new_record["sender_id"]

        # 找到最后一条同一用户的违规记录
        for i in range(len(self.moderation_records) - 1, -1, -1):
            existing_record = self.moderation_records[i]
            if (
                existing_record.get("group_id") == group_id
                and existing_record.get("sender_id") == sender_id
                and existing_record.get("is_violation")
            ):
                # 合并记录
                merged_record = existing_record.copy()

                # 更新计数
                if "violation_count" not in merged_record:
                    merged_record["violation_count"] = 1
                merged_record["violation_count"] += 1

                # 更新时间戳
                merged_record["timestamp"] = new_record["timestamp"]

                # 更新消息列表
                if "messages" not in merged_record:
                    merged_record["messages"] = [merged_record["message"]]
                merged_record["messages"].append(new_record["message"])

                # 更新消息显示为合并消息
                merged_record["message"] = (
                    f"[重复违规 {merged_record['violation_count']}次] {new_record['message']}"
                )

                # 替换原记录
                self.moderation_records[i] = merged_record

                return merged_record

        return new_record

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """监听群消息"""
        try:
            # 检查是否应该审核
            if not self._should_moderate(event):
                return

            # 存储消息上下文
            self._store_message_context(event)

            # 获取消息内容
            message_str = event.message_str
            message_chain = event.message_obj.message

            # 检查是否包含图片
            has_image = any(
                hasattr(msg, "type")
                and hasattr(msg.type, "name")
                and msg.type.name == "Image"
                for msg in message_chain
            )
            has_text = bool(message_str.strip())

            # 检查是否启用审核
            enable_text = self.config.get("enable_text_moderation", True)
            enable_image = self.config.get("enable_image_moderation", True)

            if not has_text and not has_image:
                return

            if has_text and not enable_text:
                return

            if has_image and not enable_image:
                return

            # 审核文字消息
            if has_text and enable_text:
                moderation_result = await self._moderate_message(
                    event, message_str, is_image=False
                )

                # 记录审核结果
                record = {
                    "id": f"{event.message_obj.message_id}_{datetime.now().timestamp()}",
                    "group_id": event.get_group_id(),
                    "sender_id": event.get_sender_id(),
                    "sender_name": event.get_sender_name(),
                    "message": message_str,
                    "message_type": "text",
                    "context": self._get_message_context(event),
                    "ai_result": moderation_result,
                    "is_violation": moderation_result.get("is_violation", False),
                    "reason": moderation_result.get("reason", ""),
                    "severity": moderation_result.get("severity", 3),
                    "timestamp": datetime.now().isoformat(),
                    "admin_reviewed": False,
                    "admin_correct": None,
                    "admin_comment": "",
                }

                # 检查是否应该合并记录
                if self._should_merge_record(record):
                    logger.info(f"合并违规记录: 用户 {record['sender_name']}")
                    self._merge_record(record)
                else:
                    self.moderation_records.append(record)

                self._save_moderation_records()

                # 处理违规消息
                if moderation_result.get("is_violation"):
                    await self._handle_violation(event, moderation_result)

            # 审核图片消息
            if has_image and enable_image:
                for msg in message_chain:
                    if (
                        hasattr(msg, "type")
                        and hasattr(msg.type, "name")
                        and msg.type.name == "Image"
                    ):
                        image_url = (
                            msg.url
                            if hasattr(msg, "url")
                            else (msg.path if hasattr(msg, "path") else None)
                        )

                        if not image_url:
                            logger.warning(f"图片没有URL或path，跳过审核")
                            continue

                        logger.info(f"开始审核图片: {image_url}")

                        # 首先尝试OCR识别图片中的文字
                        ocr_text = await self._extract_text_from_image_url(image_url)
                        if ocr_text:
                            logger.info(f"OCR识别到文字: {ocr_text}")
                            # 对OCR识别的文字进行本地关键词检查
                            local_result = self._check_local_keywords(ocr_text)
                            if local_result.get("is_violation"):
                                logger.info(f"图片中的文字触发违规关键词，直接判定违规")
                                moderation_result = local_result
                            else:
                                # 没有触发关键词，继续AI审核
                                moderation_result = (
                                    await self._moderate_image_with_retry(
                                        event, image_url
                                    )
                                )
                        else:
                            # OCR失败，直接AI审核
                            moderation_result = await self._moderate_image_with_retry(
                                event, image_url
                            )

                        # 记录审核结果
                        record = {
                            "id": f"{event.message_obj.message_id}_{datetime.now().timestamp()}",
                            "group_id": event.get_group_id(),
                            "sender_id": event.get_sender_id(),
                            "sender_name": event.get_sender_name(),
                            "message": f"[图片] {image_url}",
                            "message_type": "image",
                            "context": self._get_message_context(event),
                            "ai_result": moderation_result,
                            "is_violation": moderation_result.get(
                                "is_violation", False
                            ),
                            "reason": moderation_result.get("reason", ""),
                            "severity": moderation_result.get("severity", 3),
                            "timestamp": datetime.now().isoformat(),
                            "admin_reviewed": False,
                            "admin_correct": None,
                            "admin_comment": "",
                        }

                        # 检查是否应该合并记录
                        if self._should_merge_record(record):
                            logger.info(f"合并违规记录: 用户 {record['sender_name']}")
                            self._merge_record(record)
                        else:
                            self.moderation_records.append(record)

                        self._save_moderation_records()

                        # 处理违规消息
                        if moderation_result.get("is_violation"):
                            await self._handle_violation(event, moderation_result)

                        # 处理审核失败的情况
                        elif moderation_result.get("error_type") == "network_error":
                            # 网络错误：开启全体禁言并@管理员
                            await self._handle_network_error(event, moderation_result)

                        elif moderation_result.get("error_type") == "safety_check":
                            # 安全检查拦截：直接撤回并禁言
                            await self._handle_violation(event, moderation_result)

                        break  # 只审核第一张图片

            # 审核合并转发消息
            if enable_image or enable_text:
                for msg in message_chain:
                    if (
                        hasattr(msg, "type")
                        and hasattr(msg.type, "name")
                        and msg.type.name == "Forward"
                    ):
                        logger.info("检测到合并转发消息，开始审核")
                        await self._moderate_forward_message(event, msg)

        except Exception as e:
            logger.error(f"处理群消息失败: {e}")

    async def _handle_violation(self, event: AstrMessageEvent, moderation_result: Dict):
        """处理违规消息"""
        try:
            group_id = event.get_group_id()
            sender_id = event.get_sender_id()
            sender_name = event.get_sender_name()
            message = event.message_str
            reason = moderation_result.get("reason", "消息违规")
            severity = moderation_result.get("severity", 3)

            logger.info(f"开始处理违规消息: 原因={reason}, 严重程度={severity}")

            # 刷屏检测
            violation_key = (group_id, sender_id)
            current_time = datetime.now()

            # 初始化或获取违规记录
            if violation_key not in self.spam_violations:
                self.spam_violations[violation_key] = {
                    "count": 0,
                    "first_time": current_time,
                    "last_time": current_time,
                    "messages": [],
                }

            # 更新违规计数
            self.spam_violations[violation_key]["count"] += 1
            self.spam_violations[violation_key]["last_time"] = current_time
            self.spam_violations[violation_key]["messages"].append(
                {
                    "message": message,
                    "time": current_time,
                    "reason": reason,
                    "severity": severity,
                }
            )

            violation_count = self.spam_violations[violation_key]["count"]
            first_time = self.spam_violations[violation_key]["first_time"]
            time_diff = (current_time - first_time).total_seconds()

            logger.info(
                f"用户 {sender_name} 违规次数: {violation_count}, 时间间隔: {time_diff}秒"
            )

            # 刷屏判断：60秒内违规超过5次，直接踢人
            spam_threshold = self.config.get("spam_threshold", 5)  # 默认5次
            spam_time_window = self.config.get("spam_time_window", 60)  # 默认60秒
            enable_spam_kick = self.config.get("enable_spam_kick", True)  # 默认启用踢人

            is_spam = (
                violation_count >= spam_threshold and time_diff <= spam_time_window
            )

            if is_spam and enable_spam_kick:
                logger.warning(
                    f"检测到刷屏行为！用户 {sender_name} 在 {time_diff:.1f}秒内违规 {violation_count} 次，将踢出群聊"
                )

                # 撤回所有违规消息
                if self.config.get("auto_revoke", True):
                    for msg_record in self.spam_violations[violation_key]["messages"]:
                        try:
                            # 这里需要记录消息ID来撤回，但由于我们没有存储消息ID，暂时跳过
                            pass
                        except Exception as e:
                            logger.error(f"撤回消息失败: {e}")

                # 踢出群聊
                await self._kick_user(event)

                # 发送通知
                notification = f"⚠️ 检测到刷屏违规行为\n"
                notification += f"用户：{sender_name}\n"
                notification += f"违规次数：{violation_count} 次\n"
                notification += f"时间间隔：{time_diff:.1f} 秒\n"
                notification += f"已踢出群聊"

                await event.send(event.plain_result(notification))

                # 清除该用户的违规记录
                del self.spam_violations[violation_key]
                return

            # 正常违规处理
            # 根据违规程度设置禁言时长
            mute_durations = {
                1: 1,  # 1级：1分钟
                2: 3,  # 2级：3分钟
                3: 5,  # 3级：5分钟
                4: 10,  # 4级：10分钟
                5: 30,  # 5级：30分钟
            }
            mute_duration = mute_durations.get(severity, 5)

            # 撤回消息
            if self.config.get("auto_revoke", True):
                logger.info("开始撤回消息")
                await self._revoke_message(event)

            # 禁言用户
            if self.config.get("auto_mute", True):
                logger.info(f"开始禁言用户 {mute_duration} 分钟")
                await self._mute_user(event, mute_duration)

            # 发送通知
            # 清理reason，移除敏感词汇并格式化
            sanitized_reason = self._sanitize_reason(reason)

            notification = f"⚠️ 消息审核违规\n"
            notification += f"{sanitized_reason}\n"
            notification += f"【严重程度】{severity}/5\n"

            if self.config.get("auto_mute", True):
                notification += f"【处理结果】已禁言 {mute_duration} 分钟"

            logger.info(
                f"发送违规通知（原reason: {reason}, 清理后: {sanitized_reason}）"
            )
            await event.send(event.plain_result(notification))

        except Exception as e:
            logger.error(f"处理违规消息失败: {e}")
            # 尝试发送错误通知
            try:
                await event.send(event.plain_result(f"处理违规消息时出错: {str(e)}"))
            except:
                pass

    async def _handle_network_error(self, event: AstrMessageEvent, error_result: Dict):
        """处理网络错误：开启全体禁言并@管理员"""
        try:
            reason = error_result.get("reason", "网络错误")
            logger.warning(f"检测到网络错误，开启全体禁言: {reason}")

            # 检查是否启用全体禁言
            if not self.config.get("enable_group_ban_on_network_error", True):
                logger.info("网络错误时全体禁言功能已禁用")
                await event.send(
                    event.plain_result(
                        f"⚠️ 图片审核网络错误\n原因：{reason}\n请管理员检查网络连接"
                    )
                )
                return

            # 开启全体禁言
            if event.get_platform_name() == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                    AiocqhttpMessageEvent,
                )

                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    payloads = {
                        "group_id": event.get_group_id(),
                        "enable": True,
                    }
                    await client.api.call_action("set_group_whole_ban", **payloads)
                    logger.info(f"已开启全体禁言: {event.get_group_id()}")

            # 获取管理员列表
            admin_users = self.config.get("admin_users", [])

            # 构建通知消息
            notification = f"⚠️ 图片审核网络错误\n"
            notification += f"原因：{reason}\n"
            notification += f"已开启全体禁言\n\n"

            if admin_users:
                notification += "请管理员联系确认：\n"
                for admin_id in admin_users[:3]:  # @最多3个管理员
                    notification += f"@{admin_id} "

            notification += "\n\n管理员确认后请手动解除全体禁言"

            await event.send(event.plain_result(notification))

        except Exception as e:
            logger.error(f"处理网络错误失败: {e}")

    async def _moderate_forward_message(self, event: AstrMessageEvent, forward_msg):
        """审核合并转发消息"""
        try:
            # 获取合并消息ID
            forward_id = None
            if hasattr(forward_msg, "id"):
                forward_id = forward_msg.id
            elif hasattr(forward_msg, "data"):
                forward_id = (
                    forward_msg.data.get("id")
                    if isinstance(forward_msg.data, dict)
                    else None
                )

            if not forward_id:
                logger.warning("无法获取合并消息ID")
                return

            logger.info(f"开始审核合并消息: {forward_id}")

            # 调用API获取合并消息内容
            if event.get_platform_name() == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                    AiocqhttpMessageEvent,
                )

                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    forward_data = await client.api.call_action(
                        "get_forward_msg", id=forward_id
                    )

                    if not forward_data or "messages" not in forward_data:
                        logger.warning("获取合并消息内容失败")
                        return

                    # 遍历合并消息中的每条消息
                    for message_node in forward_data["messages"]:
                        sender_name = message_node.get("sender", {}).get(
                            "nickname", "未知用户"
                        )
                        raw_content = message_node.get("message") or message_node.get(
                            "content", []
                        )

                        # 解析消息内容
                        content_chain = []
                        if isinstance(raw_content, str):
                            try:
                                content_chain = json.loads(raw_content)
                            except:
                                content_chain = [
                                    {"type": "plain", "data": {"text": raw_content}}
                                ]
                        elif isinstance(raw_content, list):
                            content_chain = raw_content

                        # 提取文字和图片
                        text_content = ""
                        image_urls = []

                        for component in content_chain:
                            if isinstance(component, dict):
                                comp_type = component.get("type", "")
                                comp_data = component.get("data", {})

                                if comp_type == "text" or comp_type == "plain":
                                    text_content += comp_data.get("text", "")
                                elif comp_type == "image":
                                    # 检查URL或文件
                                    url = comp_data.get("url") or comp_data.get("file")
                                    if url:
                                        image_urls.append(url)
                                        logger.info(f"合并消息中发现图片: {url}")

                        # 审核文字内容
                        enable_text = self.config.get("enable_text_moderation", True)
                        if text_content and enable_text:
                            logger.info(f"审核合并消息文字: {text_content[:50]}...")
                            # 构造临时event用于审核
                            temp_event = event
                            moderation_result = await self._moderate_message(
                                temp_event, text_content, is_image=False
                            )

                            if moderation_result.get("is_violation"):
                                logger.info(
                                    f"合并消息文字违规: {moderation_result.get('reason')}"
                                )
                                await self._handle_violation(event, moderation_result)
                                return  # 发现违规，停止审核

                        # 审核图片
                        enable_image = self.config.get("enable_image_moderation", True)
                        if image_urls and enable_image:
                            for image_url in image_urls:
                                logger.info(f"审核合并消息图片: {image_url}")

                                # 尝试审核图片
                                moderation_result = None
                                for retry in range(3):
                                    try:
                                        moderation_result = (
                                            await self._moderate_message(
                                                event, image_url, is_image=True
                                            )
                                        )
                                        if moderation_result and (
                                            moderation_result.get("is_violation")
                                            or not moderation_result.get("is_violation")
                                        ):
                                            break
                                        await asyncio.sleep(1)
                                    except Exception as e:
                                        logger.error(
                                            f"合并消息图片审核失败 (重试 {retry + 1}/3): {e}"
                                        )
                                        await asyncio.sleep(1)

                                if moderation_result and moderation_result.get(
                                    "is_violation"
                                ):
                                    logger.info(
                                        f"合并消息图片违规: {moderation_result.get('reason')}"
                                    )
                                    await self._handle_violation(
                                        event, moderation_result
                                    )
                                    return  # 发现违规，停止审核

                    logger.info("合并消息审核完成，未发现违规")

        except Exception as e:
            logger.error(f"审核合并消息失败: {e}")

    @filter.command("审核记录", alias=["审核历史", "moderation_log"])
    async def moderation_log(self, event: AstrMessageEvent):
        """查看审核记录"""
        try:
            # 获取最近的审核记录
            recent_records = self.moderation_records[-10:]

            if not recent_records:
                yield event.plain_result("暂无审核记录")
                return

            # 构建响应
            response = "📋 最近10条审核记录：\n\n"
            for i, record in enumerate(recent_records, 1):
                status = "❌ 违规" if record.get("is_violation") else "✅ 通过"
                response += f"{i}. {status}\n"
                response += f"   时间：{record.get('timestamp', '')[:19]}\n"
                response += f"   用户：{record.get('sender_name', '')}\n"
                response += f"   消息：{record.get('message', '')[:30]}...\n"
                if record.get("is_violation"):
                    response += f"   原因：{record.get('reason', '')}\n"
                response += "\n"

            yield event.plain_result(response)

        except Exception as e:
            logger.error(f"查看审核记录失败: {e}")
            yield event.plain_result("❌ 查看审核记录失败")

    @filter.command("审核统计", alias=["moderation_stats"])
    async def moderation_stats(self, event: AstrMessageEvent):
        """查看审核统计"""
        try:
            total = len(self.moderation_records)
            violations = sum(
                1 for r in self.moderation_records if r.get("is_violation")
            )
            reviewed = sum(
                1 for r in self.moderation_records if r.get("admin_reviewed")
            )
            correct = sum(1 for r in self.moderation_records if r.get("admin_correct"))

            response = "📊 审核统计信息：\n\n"
            response += f"总审核数：{total}\n"
            response += f"违规数：{violations}\n"
            response += f"通过数：{total - violations}\n"
            response += f"已复核：{reviewed}\n"
            response += f"准确率：{round(correct / reviewed * 100, 2) if reviewed > 0 else 0}%\n\n"

            response += "📚 学习数据：\n"
            response += f"正确样本：{len(self.learning_data['correct'])}\n"
            response += f"错误样本：{len(self.learning_data['incorrect'])}\n"

            yield event.plain_result(response)

        except Exception as e:
            logger.error(f"查看审核统计失败: {e}")
            yield event.plain_result("❌ 查看审核统计失败")

    @filter.command("审核管理", alias=["moderation_admin"])
    async def moderation_admin(self, event: AstrMessageEvent):
        """审核管理帮助"""
        try:
            port = self.config.get("webui_port", 8899)

            response = "🔧 AI消息审核管理\n\n"
            response += "可用命令：\n"
            response += "/审核记录 - 查看最近审核记录\n"
            response += "/审核统计 - 查看审核统计信息\n"
            response += f"WebUI管理：http://你的服务器地址:{port}\n\n"
            response += "请在WebUI中进行详细的管理和复核操作"

            yield event.plain_result(response)

        except Exception as e:
            logger.error(f"审核管理失败: {e}")
            yield event.plain_result("❌ 审核管理失败")

    async def terminate(self):
        """插件卸载时调用"""
        try:
            if self.webui_runner:
                await self.webui_runner.cleanup()
            logger.info("AI消息审核插件已卸载")
        except Exception as e:
            logger.error(f"卸载插件时出错: {e}")
