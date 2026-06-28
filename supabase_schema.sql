-- OpClave / Council — Supabase schema
-- Run this in the Supabase dashboard SQL editor (Database → SQL Editor → New query).
-- Safe to re-run: uses CREATE TABLE IF NOT EXISTS and CREATE POLICY IF NOT EXISTS.
-- If tables already exist with mismatched schema, drop them first:
--   DROP TABLE IF EXISTS public.threads;
--   DROP TABLE IF EXISTS public.artifacts;

-- ── threads ────────────────────────────────────────────────────────────────────
-- id is a client-generated string like "thread-1718000000000-abc12" (not a UUID).
create table if not exists public.threads (
  id            text        primary key,
  user_id       uuid        not null references auth.users(id) on delete cascade,
  name          text        not null default 'New Thread',
  history       jsonb       not null default '[]',
  messages_html text        not null default '',
  mode          text        not null default 'chat',
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

-- ── artifacts ─────────────────────────────────────────────────────────────────
-- id is Supabase auto-generated UUID (returned to client as cloudId).
create table if not exists public.artifacts (
  id              uuid        primary key default gen_random_uuid(),
  user_id         uuid        not null references auth.users(id) on delete cascade,
  name            text        not null default 'Untitled',
  type            text        not null default 'research',
  question        text,
  synthesis       text,
  model_responses jsonb       not null default '{}',
  web_sources     jsonb       not null default '[]',
  papers          jsonb       not null default '[]',
  tags            jsonb       not null default '[]',
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

-- ── Row Level Security ─────────────────────────────────────────────────────────
alter table public.threads   enable row level security;
alter table public.artifacts enable row level security;

-- Each user can only read/write their own rows.
do $$ begin
  if not exists (
    select 1 from pg_policies
    where tablename = 'threads' and policyname = 'Users manage own threads'
  ) then
    create policy "Users manage own threads"
      on public.threads for all
      using (auth.uid() = user_id)
      with check (auth.uid() = user_id);
  end if;
end $$;

do $$ begin
  if not exists (
    select 1 from pg_policies
    where tablename = 'artifacts' and policyname = 'Users manage own artifacts'
  ) then
    create policy "Users manage own artifacts"
      on public.artifacts for all
      using (auth.uid() = user_id)
      with check (auth.uid() = user_id);
  end if;
end $$;

-- ── Helper: delete_user RPC (called by AuthManager.deleteAccount) ─────────────
-- Allows a user to delete their own auth account from the client side.
create or replace function public.delete_user()
returns void
language plpgsql
security definer
as $$
begin
  delete from auth.users where id = auth.uid();
end;
$$;
