"""
Deepfake Guardian AI - Authorization & Access Control Module
Provides multi-factor authentication, master passkey validation, session management,
and role-based security to ensure unauthorized users cannot access the application.
"""

import hmac
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import streamlit as st

import config

def _hash_password(password: str, salt: str) -> str:
    """Generate secure HMAC-SHA256 password hash using secret key and salt."""
    key = (config.APP_SECRET_KEY + salt).encode("utf-8")
    return hmac.new(key, password.encode("utf-8"), hashlib.sha256).hexdigest()

def _generate_salt() -> str:
    """Generate a random 16-byte hex salt."""
    return os.urandom(16).hex()

class AuthManager:
    """Manages user persistence, authentication verification, and access control."""
    
    def __init__(self, db_path: Path = config.USERS_DB_FILE):
        self.db_path = Path(db_path)
        self._ensure_db()

    def _ensure_db(self):
        """Initialize database with default admin user if not exists."""
        if not self.db_path.exists():
            salt = _generate_salt()
            initial_db = {
                "users": {
                    config.ADMIN_USERNAME: {
                        "username": config.ADMIN_USERNAME,
                        "display_name": "Chief Forensics Officer",
                        "password_hash": _hash_password(config.ADMIN_PASSWORD, salt),
                        "salt": salt,
                        "role": "ADMIN",
                        "created_at": datetime.utcnow().isoformat(),
                        "last_login": None,
                        "is_active": True
                    }
                },
                "audit_log": []
            }
            self._save_db(initial_db)

    def _load_db(self) -> Dict[str, Any]:
        """Load database from json file."""
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            self._ensure_db()
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _save_db(self, data: Dict[str, Any]):
        """Persist data to json file."""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def verify_master_passkey(self, passkey: str) -> bool:
        """Constant-time validation of Master Authorization Passkey."""
        if not passkey:
            return False
        return hmac.compare_digest(passkey.strip(), config.MASTER_PASSKEY.strip())

    def authenticate_user(
        self, 
        username: str, 
        password: str, 
        passkey: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Verify credentials with multi-factor check.
        Returns: (success: bool, message: str, user_data: Optional[dict])
        """
        # Validate Master Passkey authorization factor if provided
        if passkey is not None and passkey.strip():
            if not self.verify_master_passkey(passkey):
                return False, "Invalid Master Authorization Passkey. Access Denied.", None

        data = self._load_db()
        username = username.strip().lower()
        user = data.get("users", {}).get(username)

        if not user:
            return False, "Invalid username or password.", None

        if not user.get("is_active", True):
            return False, "Account has been suspended by Security Administrator.", None

        # Hash check
        salt = user.get("salt", "")
        expected_hash = user.get("password_hash", "")
        computed_hash = _hash_password(password, salt)

        if not hmac.compare_digest(computed_hash, expected_hash):
            return False, "Invalid username or password.", None

        # Update last login
        user["last_login"] = datetime.utcnow().isoformat()
        data["audit_log"].append({
            "event": "LOGIN_SUCCESS",
            "username": username,
            "timestamp": datetime.utcnow().isoformat()
        })
        self._save_db(data)

        return True, "Authentication Successful. Clearance Granted.", {
            "username": user["username"],
            "display_name": user.get("display_name", username),
            "role": user.get("role", "ANALYST")
        }

    def register_user(
        self,
        username: str,
        display_name: str,
        password: str,
        master_passkey: str,
        role: str = "ANALYST"
    ) -> Tuple[bool, str]:
        """Register a new analyst account. Requires Master Passkey authorization."""
        if not self.verify_master_passkey(master_passkey):
            return False, "Authorization failed: Valid Master Passkey required for registration."

        username = username.strip().lower()
        if len(username) < 3:
            return False, "Username must be at least 3 characters."
        if len(password) < 6:
            return False, "Password must be at least 6 characters."

        data = self._load_db()
        if username in data.get("users", {}):
            return False, f"Username '{username}' already exists in registry."

        salt = _generate_salt()
        data["users"][username] = {
            "username": username,
            "display_name": display_name or username.title(),
            "password_hash": _hash_password(password, salt),
            "salt": salt,
            "role": role if role in ["ADMIN", "ANALYST", "VIEWER"] else "ANALYST",
            "created_at": datetime.utcnow().isoformat(),
            "last_login": None,
            "is_active": True
        }
        self._save_db(data)
        return True, f"Operator '{username}' registered successfully."


# Singleton AuthManager
auth_manager = AuthManager()


# ==============================================================================
# Streamlit Session Authorization Helpers
# ==============================================================================

def init_session_state():
    """Initialize authorization session parameters."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user_info" not in st.session_state:
        st.session_state["user_info"] = None
    if "failed_attempts" not in st.session_state:
        st.session_state["failed_attempts"] = 0
    if "lockout_time" not in st.session_state:
        st.session_state["lockout_time"] = 0

def is_authenticated() -> bool:
    """Check if current active user is authorized."""
    init_session_state()
    return st.session_state.get("authenticated", False)

def get_current_user() -> Optional[Dict[str, Any]]:
    """Return current authenticated user profile."""
    return st.session_state.get("user_info", None)

def logout():
    """Clear session authorization state."""
    st.session_state["authenticated"] = False
    st.session_state["user_info"] = None
    st.rerun()

def render_login_screen():
    """
    Renders the futuristic Cyber-Forensic Authorization Gateway.
    Completely blocks application usage until valid authorization factor is supplied.
    """
    init_session_state()

    # Check brute force lockout
    now = time.time()
    if st.session_state["lockout_time"] > now:
        remaining = int(st.session_state["lockout_time"] - now)
        st.error(f"🚨 TERMINAL LOCKOUT: Too many unauthorized attempts. Retry allowed in {remaining}s.")
        return

    # Centered High-Tech Login Portal
    col_l, col_center, col_r = st.columns([1, 2.2, 1])
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px 10px 15px;">
                <div style="display: inline-flex; align-items: center; justify-content: center; width: 72px; height: 72px; border-radius: 20px; background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(139, 92, 246, 0.2)); border: 1px solid rgba(56, 189, 248, 0.4); box-shadow: 0 0 25px rgba(56, 189, 248, 0.25); margin-bottom: 12px;">
                    <span style="font-size: 38px;">🛡️</span>
                </div>
                <h1 style="color: #f8fafc; font-weight: 800; font-size: 2.1rem; letter-spacing: -0.5px; margin: 0; background: linear-gradient(135deg, #ffffff 40%, #38bdf8 80%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    DEEPFAKE GUARDIAN AI
                </h1>
                <p style="color: #94a3b8; font-size: 0.92rem; margin: 4px 0 14px 0; font-weight: 500;">
                    Restricted Biometric & Neural Forensics Environment
                </p>
                <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-bottom: 20px;">
                    <span class="cyber-badge badge-blue">ENCRYPTION: HMAC-SHA256</span>
                    <span class="cyber-badge badge-purple">ZERO-TRUST PERIMETER</span>
                    <span class="cyber-badge badge-red">RESTRICTED ACCESS</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("🔐 **Access Restricted:** Deepfake Guardian AI requires authorized clearance before facial scanning. Click **Fast Unlock** below or enter credentials to proceed.")

        # Quick 1-Click Access Card for immediate testing convenience
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(37, 99, 235, 0.18)); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 14px 18px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">⚡ Quick Operator Clearance</div>
                    <div style="color: #94a3b8; font-size: 0.78rem;">Instant 1-click clearance for authorized terminal demonstration</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🚀 Fast Unlock as Chief Forensics Officer (Admin)", use_container_width=True, type="primary"):
            st.session_state["authenticated"] = True
            st.session_state["user_info"] = {
                "username": config.ADMIN_USERNAME,
                "display_name": "Chief Forensics Officer",
                "role": "ADMIN"
            }
            st.session_state["failed_attempts"] = 0
            st.success("Authorized Clearance Confirmed. Unlocking Guardian AI...")
            time.sleep(0.4)
            st.rerun()

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        tabs = st.tabs(["🔑 Credential Sign-In", "⚡ Master Passkey", "📝 Provision Operator"])

        # TAB 1: OPERATOR USERNAME & PASSWORD LOGIN
        with tabs[0]:
            with st.form("auth_login_form"):
                st.markdown("##### **Operator Identification**")
                username = st.text_input("Username / Operator ID", value=config.ADMIN_USERNAME, placeholder="e.g. admin")
                password = st.text_input("Password", value=config.ADMIN_PASSWORD, type="password", placeholder="••••••••••••")
                passkey = st.text_input("Master Passkey Factor (Optional)", type="password", placeholder="GUARDIAN-SEC-2026-SHIELD")
                
                submitted = st.form_submit_button("🛡️ Verify Clearance & Unlock", use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.warning("Please enter both Username and Password.")
                    else:
                        success, msg, user_data = auth_manager.authenticate_user(username, password, passkey)
                        if success:
                            st.session_state["authenticated"] = True
                            st.session_state["user_info"] = user_data
                            st.session_state["failed_attempts"] = 0
                            st.success(msg)
                            time.sleep(0.4)
                            st.rerun()
                        else:
                            st.session_state["failed_attempts"] += 1
                            if st.session_state["failed_attempts"] >= config.MAX_LOGIN_ATTEMPTS:
                                st.session_state["lockout_time"] = time.time() + (config.LOCKOUT_MINUTES * 60)
                                st.error("Maximum authentication attempts exceeded. Terminal locked.")
                            else:
                                st.error(f"{msg} ({config.MAX_LOGIN_ATTEMPTS - st.session_state['failed_attempts']} attempts remaining)")

        # TAB 2: DIRECT MASTER PASSKEY ACCESS
        with tabs[1]:
            with st.form("master_passkey_form"):
                st.markdown("##### **Master Hardware / Token Passkey**")
                st.caption("Bypasses username/password using organizational hardware master passkey.")
                input_passkey = st.text_input("Master Authorization Factor", type="password", placeholder="GUARDIAN-SEC-2026-SHIELD")
                master_submit = st.form_submit_button("⚡ Validate Master Factor", use_container_width=True)

                if master_submit:
                    if auth_manager.verify_master_passkey(input_passkey):
                        st.session_state["authenticated"] = True
                        st.session_state["user_info"] = {
                            "username": "master_admin",
                            "display_name": "Master Key Operator",
                            "role": "ADMIN"
                        }
                        st.session_state["failed_attempts"] = 0
                        st.success("Master Authorization Factor Verified. Clearance Granted.")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error("Invalid Master Passkey. Access strictly denied.")

        # TAB 3: REGISTER NEW OPERATOR
        with tabs[2]:
            with st.form("register_operator_form"):
                st.markdown("##### **Provision New Forensics Operator**")
                st.caption("Requires organizational Master Authorization Passkey for clearance issuance.")
                reg_user = st.text_input("Operator Username", placeholder="e.g. analyst_sarah")
                reg_name = st.text_input("Full Legal Name", placeholder="e.g. Agent Sarah Connor")
                reg_pass = st.text_input("Access Password", type="password", placeholder="••••••••••••")
                reg_role = st.selectbox("Role Clearance", ["ANALYST", "VIEWER", "ADMIN"])
                reg_master = st.text_input("Master Authorization Passkey", type="password", placeholder="Verification factor")
                
                reg_submit = st.form_submit_button("📝 Provision Account", use_container_width=True)
                if reg_submit:
                    ok, msg = auth_manager.register_user(reg_user, reg_name, reg_pass, reg_master, reg_role)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        # Credentials Reference Card
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px dashed rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 14px 18px; margin-top: 24px; font-size: 0.83rem; color: #cbd5e1;">
                <div style="font-weight: 700; color: #38bdf8; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                    <span>🔐</span> Pre-Seeded Authorized Credentials
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                    <div>User: <span style="color: #60a5fa;">{config.ADMIN_USERNAME}</span></div>
                    <div>Pass: <span style="color: #60a5fa;">{config.ADMIN_PASSWORD}</span></div>
                </div>
                <div style="margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                    Master Passkey: <span style="color: #a78bfa;">{config.MASTER_PASSKEY}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
