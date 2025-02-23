export interface User {
    id: string;
    email: string;
    username: string;
    is_active: boolean;
    is_2fa_enabled: boolean;
    created_at: string;
    last_login?: string;
}

export interface UserCreate {
    email: string;
    username: string;
    password: string;
}

export interface LoginResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    user: User;
}

export interface TwoFactorSetup {
    secret: string;
    qr_code: string;
    backup_codes: string[];
}

export interface TwoFactorVerify {
    token: string;
}
